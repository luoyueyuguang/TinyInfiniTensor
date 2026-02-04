#include "core/graph.h"
#include <algorithm>
#include <numeric>
#include <queue>
//引入reinterpret_cast
#include <cstdint>
namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
      // 第一遍处理Transpose合并
for (size_t i = 0; i < ops.size(); i++) {
    auto op = ops[i];
    
    if (op->getOpType() == OpType::Transpose) {
        auto op_transpose = std::dynamic_pointer_cast<TransposeObj>(op);
        auto input = op_transpose->getInputs(0);
        auto preOp = input->getSource();
        
        if (preOp && preOp->getOpType() == OpType::Transpose && input->targets.size() == 1) {
            auto preOp_transpose = std::dynamic_pointer_cast<TransposeObj>(preOp);
            auto preInput = preOp_transpose->getInputs(0);
            auto perm_op1 = op_transpose->getPermute();
            auto perm_op2 = preOp_transpose->getPermute();
            
            // 计算合并后的permute
            std::vector<int> merged_perm = perm_op1;
            bool merge_flag = true;
            for (size_t m = 0; m < merged_perm.size(); m++) {
                merged_perm[m] = perm_op2[perm_op1[m]];
                if (merged_perm[m] != int(m)) {
                    merge_flag = false;
                }
            }
            
            // 保存后继算子列表
            auto successors = op->getSuccessors();
            auto output_tensor = op->getOutput();
            
            // 更新输入连接
            preInput->removeTarget(preOp);
            
            if (merge_flag) {
                // 完全消除两个transpose
                for (auto succ : successors) {
                    succ->replaceInput(output_tensor, preInput);
                    preInput->addTarget(succ);
                }
                this->removeTensor(output_tensor);
            } else {
                // 合并为一个新的transpose
                auto new_op = make_ref<TransposeObj>(this, preInput, output_tensor, merged_perm);
                // 连接后继算子
                for (auto succ : successors) {
                    new_op->addSuccessors(succ);
                    succ->addPredecessors(new_op);
                }
                this->addOperatorAndConnect(new_op);
            }
            
            // 清理原算子的连接关系
            // 断开preOp与前驱的连接
            for (auto pre : preOp->getPredecessors()) {
                pre->removeSuccessors(preOp);
                preOp->removePredecessors(pre);
            }
            
            // 断开op与后继的连接
            for (auto succ : successors) {
                succ->removePredecessors(op);
                op->removeSuccessors(succ);
            }
            
            // 删除算子
            this->removeOperator(op);
            this->removeOperator(preOp);
            this->removeTensor(input);
            
            // 由于删除了两个算子，需要调整索引
            i--;
            if (i > 0) i--; // 回退两个位置
        }
    }
}

// 第二遍处理MatMul与Transpose融合
for (size_t i = 0; i < ops.size(); i++) {
    auto op = ops[i];
    
    if (op->getOpType() == OpType::MatMul) {
        auto op_mul = std::dynamic_pointer_cast<MatmulObj>(op);
        
        // 处理输入A
        auto mata = op_mul->getInputs(0);
        auto pre_op_mul_a = mata->getSource();
        if (pre_op_mul_a && pre_op_mul_a->getOpType() == OpType::Transpose && mata->targets.size() == 1) {
            auto op_transpose_a = std::dynamic_pointer_cast<TransposeObj>(pre_op_mul_a);
            auto perm = op_transpose_a->getPermute();
            bool merge_flag = true;
            
            // 检查是否是最后两维的转置
            for (size_t m = 0; m < perm.size() - 2; m++) {
                if (perm[m] != int(m)) {
                    merge_flag = false;
                    break;
                }
            }
            
            if (merge_flag && 
                perm[perm.size() - 1] == int(perm.size() - 2) && 
                perm[perm.size() - 2] == int(perm.size() - 1)) {
                
                auto transpose_input = pre_op_mul_a->getInputs(0);
                op_mul->setTransA(!op_mul->getTransA());
                
                // 保存原连接关系
                auto transpose_predecessors = pre_op_mul_a->getPredecessors();
                
                // 更新输入连接
                transpose_input->removeTarget(pre_op_mul_a);
                transpose_input->addTarget(op_mul);
                op_mul->inputs[0] = transpose_input;
                
                // 更新算子连接关系
                for (auto pre : transpose_predecessors) {
                    pre->removeSuccessors(pre_op_mul_a);
                    pre->addSuccessors(op_mul);
                    op_mul->addPredecessors(pre);
                }
                
                // 双向断开连接
                op_mul->removePredecessors(pre_op_mul_a);
                pre_op_mul_a->removeSuccessors(op_mul);
                
                // 清理资源
                this->removeTensor(mata);
                this->removeOperator(pre_op_mul_a);
                
                // 由于删除了算子，需要调整索引
                i--;
            }
        }
        
        // 处理输入B
        auto matb = op_mul->getInputs(1);
        auto pre_op_mul_b = matb->getSource();
        if (pre_op_mul_b && pre_op_mul_b->getOpType() == OpType::Transpose && matb->targets.size() == 1) {
            auto op_transpose_b = std::dynamic_pointer_cast<TransposeObj>(pre_op_mul_b);
            auto perm = op_transpose_b->getPermute();
            bool merge_flag = true;
            
            // 检查是否是最后两维的转置
            for (size_t m = 0; m < perm.size() - 2; m++) {
                if (perm[m] != int(m)) {
                    merge_flag = false;
                    break;
                }
            }
            
            if (merge_flag && 
                perm[perm.size() - 1] == int(perm.size() - 2) && 
                perm[perm.size() - 2] == int(perm.size() - 1)) {
                
                auto transpose_input = pre_op_mul_b->getInputs(0);
                op_mul->setTransB(!op_mul->getTransB());
                
                // 保存原连接关系
                auto transpose_predecessors = pre_op_mul_b->getPredecessors();
                
                // 更新输入连接
                transpose_input->removeTarget(pre_op_mul_b);
                transpose_input->addTarget(op_mul);
                op_mul->inputs[1] = transpose_input;
                
                // 更新算子连接关系
                for (auto pre : transpose_predecessors) {
                    pre->removeSuccessors(pre_op_mul_b);
                    pre->addSuccessors(op_mul);
                    op_mul->addPredecessors(pre);
                }
                
                // 双向断开连接
                op_mul->removePredecessors(pre_op_mul_b);
                pre_op_mul_b->removeSuccessors(op_mul);
                
                // 清理资源
                this->removeTensor(matb);
                this->removeOperator(pre_op_mul_b);
                
                // 由于删除了算子，需要调整索引
                i--;
            }
        }
    }
}
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        auto n = this->tensors.size();
        for (size_t i = 0; i < n; i++) {
            auto bytes = this->tensors[i]->getBytes();
            auto blob = make_ref<BlobObj>(this->runtime, reinterpret_cast<char *>(this->allocator.getPtr()) + bytes);
            this->tensors[i]->setDataBlob(blob);
        }

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini