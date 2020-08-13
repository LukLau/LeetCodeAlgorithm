package org.code.algorithm.datastructe;

/**
 * @author luk
 * @date 2020/7/18
 */
public class TreeNode {
    public int val;
    public TreeNode left;
    public TreeNode right;
    public TreeNode parent;

    public TreeNode root;

    public static final boolean RED = true;

    public static final boolean BLACK = false;

    public Boolean color = null;

    public TreeNode() {
    }

    public TreeNode(int val) {
        this.val = val;
    }

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    public void leftRotate(TreeNode node) {
        if (node == null) {
            return;
        }
        TreeNode right = node.right;

        node.right = right.left;

        right.left.parent = node;

        right.parent = node.parent;

        if (node.parent == null) {
            root = right;
        } else if (node.parent.left == node) {
            node.parent.left = right;
        } else if (node.parent.right == node) {
            node.parent.right = right;
        }
        node.parent = right;

        right.left = node;
    }

    public void rightRotate(TreeNode node) {
        if (node == null) {
            return;
        }

        TreeNode left = node.left;


        node.left = left.right;


        left.right.parent = node;

        left.parent = node.parent;

        if (node.parent == null) {
            root = left;
        } else if (node.parent.left == node) {
            node.parent.left = left;
        } else if (node.parent.right == node) {
            node.parent.right = left;
        }
        node.parent = left;

        left.right = node;
    }

}
