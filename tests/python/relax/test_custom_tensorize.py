import tvm
import tvm.testing
from tvm import relax, tir, IRModule
from tvm.ir import GlobalVar, Op
from tvm.runtime import const, convert
from tvm.script import relax as R, tir as T
from tvm.relax import Call, Expr
from tvm.relax.utils import copy_with_new_vars, normalize_expr
from tvm.relax.expr_functor import mutator, PyExprMutator
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.op.base import call_tir
from tvm.relax.transform.legalize_ops.common import TEFunc, LegalizeFunc, register_legalize
from tvm.tir.schedule.analysis import loop_domain_of_sref_tree_path
from typing import List


import os
import shutil
import pdb


@tvm.instrument.pass_instrument
class DumpIR:
    """Print the name of the pass, the IR, only before passes execute.
    with tvm.transform.PassContext(config={}, instruments=[DumpIR()]):
        mod = pass_seq(mod)
    """

    def __init__(self, root_dir: str = None):
        self.counts = 0
        if root_dir is None:
            self.root_dir = "dump_ir"
        else:
            self.root_dir = root_dir

        try:
            if os.path.exists(self.root_dir):
                shutil.rmtree(self.root_dir, ignore_errors=False)
        except EnvironmentError:
            if os.path.exists(self.root_dir):
                os.system("rm -rf " + self.root_dir)
        try:
            os.makedirs(self.root_dir, mode=0o700)
        except EnvironmentError:
            print("NOTE: CANNOT mkdir " + self.root_dir)

    def run_before_pass(self, mod, info):
        """instrument before pass"""
        pass

    def run_after_pass(self, mod, info):
        """instrument after pass"""
        pname = str(self.counts).rjust(5, "0") + "_" + info.name + ".ir"
        pname = os.path.join(self.root_dir, pname)
        with open(pname, "w", encoding="utf-8") as f:
            f.write(str(mod))
        self.counts += 1


def _binary(te_func: TEFunc) -> LegalizeFunc:
    def binary_call_te(bb: BlockBuilder, call: Call) -> Expr:
        tir_func = te_func(call)
        call_args = [x for x in call.args]
        output_sinfo = call.struct_info
        tir_vars = None

        primfunc_name = te_func.__name__
        gvar = bb.add_func(tir_func, primfunc_name)

        return call_tir(gvar, call_args, output_sinfo, tir_vars)

    return binary_call_te


def call_tir_intrin(dtype, func_name, *args, span=None):
    return tir.Call(dtype, func_name, convert(args), span)


def vec_multiply_insn(dtype, rst, rst_idx, left, left_idx, right, right_idx):
    return call_tir_intrin(
        dtype,
        "tir.evas_vec_multiply",
        rst,
        rst_idx,
        left,
        left_idx,
        right,
        right_idx,
    )


def vec_add_insn(dtype, rst, rst_idx, left, left_idx, right, right_idx):
    return call_tir_intrin(
        dtype,
        "tir.evas_vec_add",
        rst,
        rst_idx,
        left,
        left_idx,
        right,
        right_idx,
    )


def vec_multiply(call):
    lhs_shape = call.args[0].struct_info.shape.values
    rhs_shape = call.args[1].struct_info.shape.values
    rst_shape = call.struct_info.shape.values

    @T.prim_func
    def multiply(lhs: T.handle, rhs: T.handle, rst: T.handle) -> None:
        lhs_buff = T.match_buffer(lhs, lhs_shape, dtype="float32")
        rhs_buff = T.match_buffer(rhs, rhs_shape, dtype="float32")
        rst_buff = T.match_buffer(rst, rst_shape, dtype="float32")

        T.attr("outer loop", "pragma_layout", "NHWC")
        T.attr("fusion_tag", "common", "start")
        for ax0, ax1 in T.grid(rst_shape[0] // 4, 1):
            T.attr("fusion_tag", "common", "stop")
            T.attr("inner loop", "dispatch_size", "[m, n] + [1, n] -> [m, n]")
            with T.block("inner_vec_add_block"):
                T.block_attr({"tensorized": 1})
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])

                T.reads(
                    [
                        lhs_buff[v_ax0 * 4 : v_ax0 * 4 + 4, v_ax1 * lhs_shape[1] : v_ax1 * lhs_shape[1] + lhs_shape[1]],
                        rhs_buff[0 : rhs_shape[0], 0 : rhs_shape[1]],
                    ]
                )
                T.writes(
                    rst_buff[v_ax0 * 4 : v_ax0 * 4 + 4, v_ax1 * rst_shape[1] : v_ax1 * rst_shape[1] + rst_shape[1]]
                )

                strides_str = "[0, 0]"
                T.attr("insn_csr", "stride_size", strides_str)
                T.evaluate(
                    vec_multiply_insn(
                        "handle",
                        rst_buff.data,
                        tir.BufferLoad(rst_buff, [v_ax0 * 4, v_ax1]),
                        # v_ax0 * 4 * 64 + v_ax1,
                        lhs_buff.data,
                        tir.BufferLoad(lhs_buff, [v_ax0 * 4, v_ax1]),
                        # v_ax0 * 4 * 64 + v_ax1,
                        rhs_buff.data,
                        const(0, dtype="int32"),
                    )
                )

    return multiply


def vec_add(call):
    lhs_shape = call.args[0].struct_info.shape.values
    rhs_shape = call.args[1].struct_info.shape.values
    rst_shape = call.struct_info.shape.values

    @T.prim_func
    def add(lhs: T.handle, rhs: T.handle, rst: T.handle) -> None:
        lhs_buff = T.match_buffer(lhs, lhs_shape, dtype="float32")
        rhs_buff = T.match_buffer(rhs, rhs_shape, dtype="float32")
        rst_buff = T.match_buffer(rst, rst_shape, dtype="float32")

        T.attr("outer loop", "pragma_layout", "NHWC")
        T.attr("fusion_tag", "common", "start")
        for ax0, ax1 in T.grid(rst_shape[0] // 4, 1):
            T.attr("fusion_tag", "common", "stop")
            T.attr("inner loop", "dispatch_size", "[m, n] + [1, n] -> [m, n]")
            with T.block("inner_vec_add_block"):
                T.block_attr({"tensorized": 1})
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])

                T.reads(
                    [
                        lhs_buff[v_ax0 * 4 : v_ax0 * 4 + 4, v_ax1 * lhs_shape[1] : v_ax1 * lhs_shape[1] + lhs_shape[1]],
                        rhs_buff[0 : rhs_shape[0], 0 : rhs_shape[1]],
                    ]
                )
                T.writes(
                    rst_buff[v_ax0 * 4 : v_ax0 * 4 + 4, v_ax1 * rst_shape[1] : v_ax1 * rst_shape[1] + rst_shape[1]]
                )

                strides_str = "[0, 0]"
                T.attr("insn_csr", "stride_size", strides_str)
                T.evaluate(
                    vec_add_insn(
                        "handle",
                        rst_buff.data,
                        tir.BufferLoad(rst_buff, [v_ax0 * 4, v_ax1]),
                        # v_ax0 * 4 * 64 + v_ax1,
                        lhs_buff.data,
                        tir.BufferLoad(lhs_buff, [v_ax0 * 4, v_ax1]),
                        # v_ax0 * 4 * 64 + v_ax1,
                        rhs_buff.data,
                        const(0, dtype="int32"),
                    )
                )

    return add


def vec_relu(call):
    pass


# register_legalize("relax.multiply", _binary(vec_multiply))
register_legalize("relax.add", _binary(vec_add))
# register_legalize("relax.relu", _binary(vec_))


@mutator
class SpatialSplitMutator(PyExprMutator):
    """split spatial common axis across element wise ops"""

    def __init__(self, mod: IRModule = None) -> None:
        super().__init__(mod)
        self.mod = mod
        self.funcs = {}
        self.var_name_idx = {}

    def transform(self, mod: IRModule) -> IRModule:
        main_var = None
        main_func = None
        for _, gv in enumerate(mod.functions):
            self.funcs[gv] = mod[gv]
            if gv.name_hint == "main":
                main_var = gv
                main_func = mod[gv]

        assert isinstance(main_func, relax.Function)

        new_body = self.visit_expr(main_func.body)

        new_main_func = relax.Function(main_func.params, new_body, relax.ObjectStructInfo(), attrs=main_func.attrs)
        self.builder_.update_func(main_var, new_main_func)
        return self.builder_.get()

    def visit_call_(self, op) -> None:
        """do simple function split
        before:
            out = fused_func(input_a, input_b)
        after:
            (input_a_p0, input_a_p1) = split(input_a)
            out_p0 = fused_func_p0(input_a_p0, input_b)
            out_p1 = fused_func_p1(input_a_p1, input_b)
            out = (out_p0, out_p1)
        """
        new_op = super().visit_call_(op)
        if new_op.op not in self.funcs:
            return new_op

        # 32 -> 16 + 8 + 8
        inputs = R.split(new_op.args[0], 2, 0)

        in_p_0 = R.TupleGetItem(inputs, 0)
        new_attrs = {}
        if new_op.attrs:
            for i, v in enumerate(new_op.attrs):
                new_attrs[i] = v
        new_attrs["need_specialize"] = 1
        attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        out_p_0 = Call(
            new_op.op,
            [in_p_0] + new_op.args[1:],
            attrs,
            sinfo_args=[
                R.Tensor((16, 64), dtype="float32"),
                R.Tensor((1, 64), dtype="float32"),
                R.Tensor((1, 64), dtype="float32"),
            ],
        )

        in_p_1 = R.TupleGetItem(inputs, 1)
        inputs = R.split(in_p_1, 2, 0)
        outs = []
        for ax0 in range(2):
            in_p = R.TupleGetItem(inputs, ax0)
            new_attrs = {}
            if new_op.attrs:
                for i, v in enumerate(new_op.attrs):
                    new_attrs[i] = v
            new_attrs["need_specialize"] = 1
            attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
            out_p = Call(
                new_op.op,
                [in_p] + new_op.args[1:],
                attrs,
                sinfo_args=[
                    R.Tensor((8, 64), dtype="float32"),
                    R.Tensor((1, 64), dtype="float32"),
                    R.Tensor((1, 64), dtype="float32"),
                ],
            )
            outs.append(out_p)

        out_p_1 = R.concat(outs, 0)
        out_p_1.span = "TODO: make this as for loop"
        out = R.concat((out_p_0, out_p_1), 0)
        out.span = "TODO: make this as for loop"

        return out


@tvm.transform.module_pass(opt_level=0, name="SpatialSplit")
class SpatialSplit:
    """"""

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        mutator = SpatialSplitMutator(mod)
        mod = mutator.transform(mod)
        return mod


@mutator
class SpecializeMutator(PyExprMutator):
    """specialize callee functions from struct_info passed from caller"""

    def __init__(self, mod: IRModule = None) -> None:
        super().__init__(mod)
        self.mod = mod
        self.funcs = {}

    def transform(self, mod: IRModule) -> IRModule:
        main_var = None
        main_func = None
        for _, gv in enumerate(mod.functions):
            self.funcs[gv] = mod[gv]
            if gv.name_hint == "main":
                main_var = gv
                main_func = mod[gv]

        assert isinstance(main_func, relax.Function)

        new_body = self.visit_expr(main_func.body)

        new_main_func = relax.Function(main_func.params, new_body, relax.ObjectStructInfo(), attrs=main_func.attrs)
        self.builder_.update_func(main_var, new_main_func)
        return self.builder_.get()

    def visit_call_(self, op) -> None:
        """specialize call to cls.function with static shape info"""
        new_op = super().visit_call_(op)
        if new_op.op not in self.funcs:
            return new_op
        if "need_specialize" not in new_op.attrs or new_op.attrs["need_specialize"] != 1:
            return new_op

        fn = self.funcs[new_op.op]
        new_func = copy_with_new_vars(fn)
        op_args = new_op.args
        new_params = new_func.params

        assert len(op_args) == len(new_params)
        for i, _ in enumerate(op_args):
            relax.expr._reset_struct_info(new_params[i], op_args[i].struct_info)
        new_func = normalize_expr(new_func)
        new_gv = self.builder_.add_func(new_func, new_op.op.name_hint)

        new_op = relax.Call(new_gv, new_op.args, new_op.attrs, new_op.sinfo_args, new_op.span)
        return new_op


@tvm.transform.module_pass(opt_level=0, name="DoSpecialize")
class DoSpecialize:
    """"""

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        mutator = SpecializeMutator(mod)
        mod = mutator.transform(mod)
        return mod


@mutator
class LoopMergeMutator(PyExprMutator):
    """merge common outer loops across element wise computes/vector instructions,
    function same as compute_at in tir schedule
    """

    def __init__(self, mod: IRModule = None) -> None:
        super().__init__(mod)
        self.tgt_op = Op.get("relax.call_tir")
        self.funcs = {}
        self.rmap = {}

    def transform(self, mod: IRModule) -> IRModule:
        main_var = None
        main_func = None
        for _, gv in enumerate(mod.functions):
            self.funcs[gv] = mod[gv]
            if gv.name_hint == "main":
                main_var = gv
                main_func = mod[gv]

        assert isinstance(main_func, relax.Function)

        new_body = self.visit_expr(main_func.body)

        new_main_func = relax.Function(main_func.params, new_body, relax.ObjectStructInfo(), attrs=main_func.attrs)
        self.builder_.update_func(main_var, new_main_func)
        return self.builder_.get()

    def visit_call_(self, op) -> None:
        """specialize call to cls.function with static shape info"""
        new_op = super().visit_call_(op)

        if new_op.op != self.tgt_op or new_op.args[0] not in self.funcs:
            return new_op
        if new_op.args[0] in self.rmap:
            return Call(
                new_op.op, [self.rmap[new_op.op]] + new_op.args[1:], new_op.attrs, new_op.sinfo_args, new_op.span
            )

        if "fused_act" not in new_op.args[0].name_hint:
            return new_op
        fn = self.funcs[new_op.args[0]]
        if not isinstance(fn, tir.PrimFunc):
            return new_op

        sch = tir.Schedule(fn, debug_mask=1)
        root_block = sch.get_block("root")
        mul_block = sch.get_block("T_multiply")
        add_block = sch.get_block("inner_vec_add_block")
        relu_block = sch.get_block("compute")

        add_loop_domain = loop_domain_of_sref_tree_path(sch.get_sref(add_block), None)
        add_loops = sch.get_loops(add_block)

        common_fusion_loops = []
        for lp_rv in add_loops:
            lp_var = sch.get(lp_rv).loop_var
            if lp_var not in add_loop_domain:
                break
            common_fusion_loops.append((lp_rv, add_loop_domain[lp_var]))

        assert len(common_fusion_loops) != 0

        outer_factor = common_fusion_loops[0][1]
        assert outer_factor.min == 0
        outer_loop_rv = common_fusion_loops[0][0]
        add_outer, add_inner = sch.split(outer_loop_rv, [outer_factor.extent, None])

        mul_loops = sch.get_loops(mul_block)
        mul_outer, mul_inner = sch.split(mul_loops[0], [outer_factor.extent, None])

        relu_loops = sch.get_loops(relu_block)
        relu_outer, relu_inner = sch.split(relu_loops[0], [outer_factor.extent, None])

        sch.merge(mul_outer, add_outer, relu_outer)

        new_func = sch.mod["main"]
        new_gv = self.builder_.add_func(new_func, new_op.args[0].name_hint)
        self.rmap[new_op.op] = new_gv
        new_op = relax.Call(new_op.op, [new_gv] + new_op.args[1:], new_op.attrs, new_op.sinfo_args, new_op.span)

        return new_op


@tvm.transform.module_pass(opt_level=0, name="MergeCommonLoop")
class MergeCommonLoop:
    """"""

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        mutator = LoopMergeMutator(mod)
        mod = mutator.transform(mod)
        return mod


def eliminate_trival_tags(op):
    """"""

    if isinstance(op, tvm.tir.AttrStmt) and op.node in ("outer loop", "inner loop", "fusion_tag"):
        return op.body
    return None


@tvm.tir.transform.prim_func_pass(opt_level=0)
# pylint: disable = unused-argument
def TrivalTagsElimination(f, mod, ctx):
    """
    ir builder inject pass
    """

    stmt = tvm.tir.stmt_functor.ir_transform(f.body, None, eliminate_trival_tags, ["tir.AttrStmt"])
    return f.with_body(stmt)


@tvm.script.ir_module
class FusedAct:
    @R.function(private=True)
    # def fused_act(
    #     x: R.Tensor(["m", "n"], "float32"), y: R.Tensor([1, "n"], "float32"), b: R.Tensor([1, "n"], "float32")
    # ) -> R.Tensor(["m", "n"], "float32"):
    def fused_act(x: R.Tensor, y: R.Tensor, b: R.Tensor) -> R.Tensor:
        R.func_attr({"Primitive": 1})
        act = R.multiply(x, y)
        act = R.add(act, b)
        rst = R.nn.relu(act)
        return rst

    @R.function(private=False)
    def fused_norm(x: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
        R.func_attr({"Primitive": 1})
        rst = R.nn.softmax(x)
        return rst

    @R.function(pure=False)
    def main(
        x: R.Tensor((32, 64), "float32"),
        y: R.Tensor((1, 64), "float32"),
        b: R.Tensor((1, 64), "float32"),
    ):
        cls = FusedAct
        with R.dataflow():
            act: R.Tensor((32, 64), dtype="float32") = cls.fused_act(x, y, b)
            R.output(act)
        return act


def test_codegen():
    mod = FusedAct
    with tvm.transform.PassContext(config={}, instruments=[DumpIR()]):
        mod = tvm.relay.transform.InferType()(mod)
        mod = SpatialSplit()(mod)
        mod = DoSpecialize()(mod)
        mod = tvm.relay.transform.InferType()(mod)
        mod = tvm.relax.transform.DeadCodeElimination(entry_functions=["main"])(mod)

        # reference from python/tvm/relax/pipeline.py:zero_pipeline
        seq = tvm.transform.Sequential(
            [
                tvm.relax.transform.LegalizeOps(enable_warning=False),
                tvm.relax.transform.AnnotateTIROpPattern(),
                tvm.relax.transform.FoldConstant(),
                tvm.relax.transform.FuseOps(),
                tvm.relax.transform.FuseTIR(),
            ]
        )
        mod = seq(mod)

        mod = MergeCommonLoop()(mod)
        mod = TrivalTagsElimination(mod)
        mod = tvm.relax.transform.DeadCodeElimination(entry_functions=["main"])(mod)

        target = tvm.target.Target("llvm", host="llvm")
        ex = relax.build(mod, target)

    print("Hello World!")
