package org.code.algorithm.leetcode.vip;

import org.code.algorithm.datastructe.TreeNode;

import java.util.Stack;

/**
 * @author dora
 * @date 2020/8/26
 */
public class TwoPageVip {


    /**
     * 上下翻转二叉树
     *
     * @return
     */
    TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();


        while (root != null) {
            stack.push(root);
            root = root.left;
        }
        TreeNode newNode = stack.pop();

        TreeNode tmp = newNode;

        while (!stack.isEmpty()) {

            TreeNode leftNode = stack.pop();

            TreeNode parentNode = stack.peek();

            leftNode.left = parentNode.right;

            leftNode.right = parentNode;
        }
        root.left = null;

        root.right = null;
        return newNode;

    }
}
