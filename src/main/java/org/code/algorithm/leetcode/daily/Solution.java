package org.code.algorithm.leetcode.daily;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.Node;
import org.code.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

/**
 * @author dora
 * @date 2020/8/10
 */
public class Solution {


    /**
     * 696. 计数二进制子串
     *
     * @param s
     * @return
     * @date 08/10
     */
    public int countBinarySubstrings(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        List<Integer> counts = new ArrayList<>();
        int index = 0;
        while (index < s.length()) {
            char word = s.charAt(index);
            int count = 0;
            while (index < s.length() && s.charAt(index) == word) {
                index++;
                count++;
            }
            counts.add(count);
        }
        int result = 0;
        for (int i = 1; i < counts.size(); i++) {
            result += Math.min(counts.get(i - 1), counts.get(i));
        }
        return result;
    }

    public int countBinarySubstringsV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int size = s.length();
        int last = 0;
        int result = 0;

        int index = 0;
        while (index < size) {
            int count = 0;
            char word = s.charAt(index);
            while (index < size && s.charAt(index) == word) {
                index++;
                count++;
            }
            result += Math.min(last, count);
            last = count;
        }
        return result;
    }


    /**
     * todo
     * 133. 克隆图
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        return null;
    }


    /**
     * 20. 有效的括号
     *
     * @param s
     * @return
     * @date 2020/08/14
     */
    public boolean isValid(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        Stack<Character> stack = new Stack<>();
        for (char word : s.toCharArray()) {

            if (word == '{') {
                stack.push('}');
            } else if (word == '[') {
                stack.push(']');
            } else if (word == '(') {
                stack.push(')');
            } else if (stack.isEmpty()) {
                return false;
            } else {
                if (stack.peek() != word) {
                    return false;
                }
                stack.pop();
            }
        }
        return stack.isEmpty();

    }


    /**
     * 110. 平衡二叉树
     *
     * @param root
     * @return
     * @date 2020/08/17
     */
    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        int leftDepth = depth(root.left);
        int rightDepth = depth(root.right);
        if (Math.abs(leftDepth - rightDepth) > 1) {
            return false;
        }
        return isBalanced(root.left) && isBalanced(root.right);
    }

    public int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(depth(root.left), depth(root.right));
    }


    /**
     * 109. 有序链表转换二叉搜索树
     *
     * @param head
     * @return
     * @date 2020/08/19
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return intervalSortedListToBST(head, null);
    }

    private TreeNode intervalSortedListToBST(ListNode head, ListNode tail) {
        if (head == tail) {
            return null;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != tail && fast.next != tail) {
            fast = fast.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = intervalSortedListToBST(head, slow);
        root.right = intervalSortedListToBST(slow.next, tail);
        return root;
    }


    public TreeNode sortedListToBSTV2(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode middleNode = getMiddleListNode(head);
        TreeNode root = new TreeNode(middleNode.val);
        if (head == middleNode) {
            return root;
        }
        root.left = sortedListToBSTV2(head);

        root.right = sortedListToBSTV2(middleNode.next);

        return root;
    }

    private ListNode getMiddleListNode(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode prev = null;
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            prev = slow;
            fast = fast.next.next;
            slow = slow.next;
        }
        if (prev != null) {
            prev.next = null;
        }
        return slow;
    }


}
