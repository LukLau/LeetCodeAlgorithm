package org.code.algorithm.leetcode.vip;

import org.code.algorithm.datastructe.TreeNode;

import java.util.HashMap;
import java.util.Stack;

/**
 * @author dora
 * @date 2020/8/26
 */
public class TwoPageVip {


    public static void main(String[] args) {
        TwoPageVip vip = new TwoPageVip();
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);

        root.right = new TreeNode(3);

        root.left.left = new TreeNode(4);

        vip.upsideDownBinaryTreeV2(root);


    }

    /**
     * 156. 上下翻转二叉树
     *
     * @return
     */
    TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        Stack<TreeNode> stack = new Stack<>();

        TreeNode p = root;

        while (p != null) {
            stack.push(p);
            p = p.left;
        }
        TreeNode newHead = stack.peek();

        while (!stack.isEmpty()) {

            TreeNode leftNode = stack.pop();

            TreeNode parentNode = stack.isEmpty() ? null : stack.peek();

            if (parentNode != null) {
                leftNode.left = parentNode.right;

                leftNode.right = parentNode;

                parentNode.left = null;

                parentNode.right = null;
            }

        }
        return newHead;
    }

    TreeNode upsideDownBinaryTreeV2(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode leftNode = root.left;

        TreeNode node = upsideDownBinaryTreeV2(leftNode);

        leftNode.left = root.right;

        leftNode.right = root;

        root.left = null;

        root.right = null;

        return node;
    }


    /**
     * todo
     * 157 用 Read4 读取 N 个字符
     *
     * @param buf
     * @param n
     * @return
     */
    public int read(char[] buf, int n) {
        return -1;
    }

    /**
     * todo
     * 158 用 Read4 读取 N 个字符
     *
     * @param buf
     * @param n
     * @return
     */
    public int readV2(char[] buf, int n) {
        return -1;
    }


    /**
     * 159 至多包含两个不同字符的最长子串
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        // Write your code here
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        HashMap<Character, Integer> map = new HashMap<>();

        int left = -1;
        int end = 0;
        int length = s.length();
        while (end < length) {

            char word = s.charAt(end);

            map.put(word, end);

            if (map.size() > 2) {

            }
            if (end - left > result && map.size() <= 2) {
                result = Math.max(result, end - left);
            }


            end++;

        }
        return result;
    }


}
