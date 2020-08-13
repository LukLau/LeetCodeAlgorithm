package org.code.algorithm.datastructe;

import java.util.logging.XMLFormatter;
import java.util.prefs.NodeChangeEvent;

/**
 * @author dora
 * @date 2020/8/13
 */
public class RedBlackColorTree {

    private TreeNode left;

    private TreeNode right;

    private TreeNode parent;

    private TreeNode root = null;

    /*
     * 红黑树特性
     * 1. 根结点是黑色的
     * 2. 结点要么是黑色的 要么是红色的
     * 3. 所有叶子结点即null结点都是黑色的
     * 4。红色结点的子结点都是黑色的
     * 5. 从任意一个结点出发到其每个叶子结点的路径上都包含相同数量黑色结点
     *
     */

    public void put(TreeNode node) {
        if (node == null) {
            throw new NullPointerException();
        }
        if (root == null) {
            root = node;
            return;
        }
        TreeNode tmp = root;
        TreeNode prev = null;
        int diff = 0;
        while (tmp != null) {
            diff = node.val - tmp.val;
            prev = tmp;
            if (diff > 0) {
                tmp = tmp.right;
            } else if (diff < 0) {
                tmp = tmp.left;
            } else {
                tmp.val = node.val;
                return;
            }
        }
        if (diff < 0) {
            prev.left = node;
        } else {
            prev.right = node;
        }
        // todo fix
        fixAfterInsert(node);

    }


    private void fixAfterInsert(TreeNode node) {
        node.color = TreeNode.RED;
        while (node != null && node != root && node.parent.color == TreeNode.RED) {
            if (node.parent == node.parent.parent.left) {
                TreeNode right = node.parent.parent.right;
                if (right.color == TreeNode.RED) {
                    node.parent.color = TreeNode.BLACK;
                    right.color = TreeNode.BLACK;

                    node.parent.parent.color = TreeNode.BLACK;

                    node = node.parent.parent;
                } else {
                    if (node == node.parent.right) {
                        node = node.parent;
                        // left rotate
                    }
                    node.parent.color = TreeNode.BLACK;

                    node.parent.parent.color = TreeNode.RED;

                    // right rotate

                }
            }
        }
        root.color = TreeNode.BLACK;
    }


}
