package org.code.algorithm.datastructe;

/**
 * @author dora
 * @date 2020/9/16
 */
public class RedBlackTreeNode {

    private RedBlackTreeNode left;
    private RedBlackTreeNode right;
    private RedBlackTreeNode root;
    private RedBlackTreeNode parent;

    private int color = 0;

    /**
     * 红黑树五大特性
     * 1. 根结点是黑色的
     * 2. 结点要么是黑色要么是红色
     * 3. 红色结点的子结点都是黑色的
     * 4. 所有叶子结点都是黑色的
     * 5. 从一红色结点到其叶子结点的任意一条路径都包涵相同数量的黑色叶子结点
     */


    public void leftRotate(RedBlackTreeNode node) {
        if (node.right == null) {
            return;
        }
        RedBlackTreeNode rightNode = node.right;

        node.right = rightNode.left;

        if (rightNode.left != null) {
            rightNode.left.parent = node;
        }

        if (node.parent == null) {
            root = rightNode;
        } else {
            if (node.parent.left == node) {
                node.parent.left = rightNode;
            } else if (node.parent.right == node) {
                node.parent.right = right;
            }
        }
        node.parent = rightNode;

        rightNode.left = node;
    }
}
