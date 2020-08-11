package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author dora
 * @date 2020/8/10
 */
public class TwoPage {

    /**
     * 100 是否是同一课树
     *
     * @param p
     * @param q
     * @return
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }


    /**
     * 101 对称二叉树
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetric(root.left, root.right);


    }

    private boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
    }


    /**
     * 102 层次遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> linkedList = new LinkedList<>();
        List<List<Integer>> ans = new ArrayList<>();
        linkedList.offer(root);
        while (!linkedList.isEmpty()) {
            List<Integer> tmp = new ArrayList<>();
            int size = linkedList.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = linkedList.pollFirst();
                tmp.add(node.val);

                if (node.left != null) {
                    linkedList.offer(node.left);
                }
                if (node.right != null) {
                    linkedList.offer(node.right);
                }
            }
            ans.add(tmp);
        }
        return ans;
    }


    /**
     * 104. 二叉树的最大深度
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }


    /**
     * 105. 从前序与中序遍历序列构造二叉树
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null) {
            return null;
        }
        return buildTree(0, preorder, 0, inorder.length - 1, inorder);
    }

    private TreeNode buildTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inorder) {
        if (preStart >= preorder.length || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildTree(preStart + 1, preorder, inStart, index - 1, inorder);
        root.right = buildTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inorder);
        return root;
    }


    /**
     * 106. 从中序与后序遍历序列构造二叉树
     *
     * @param inorder
     * @param postorder
     * @return
     */
    public TreeNode buildTreeV2(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null) {
            return null;
        }
        return buildTreeV2(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);
    }

    private TreeNode buildTreeV2(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
        if (inStart > inEnd || postStart > postEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = 0;
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                index = i;
                break;
            }
        }
        root.left = buildTreeV2(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);
        root.right = buildTreeV2(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
        return root;
    }


    /**
     * 107. 二叉树的层次遍历 II
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<List<Integer>> result = new LinkedList<>();
        LinkedList<TreeNode> deque = new LinkedList<>();
        deque.offer(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            List<Integer> tmp = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode node = deque.pollFirst();

                tmp.add(node.val);

                if (node.left != null) {
                    deque.offer(node.left);
                }
                if (node.right != null) {
                    deque.offer(node.right);
                }
            }
            result.addFirst(tmp);
        }
        return result;
    }


    /**
     * 108. 将有序数组转换为二叉搜索树
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid - 1);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }


    /**
     * 109. 有序链表转换二叉搜索树
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        return sortedListToBST(head, null);
    }

    private TreeNode sortedListToBST(ListNode start, ListNode end) {
        if (start == end) {
            return null;
        }
        ListNode fast = start;
        ListNode slow = start;
        while (fast != end && fast.next != end) {
            slow = slow.next;
            fast = fast.next.next;
        }
        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(start, slow);
        root.right = sortedListToBST(slow.next, end);
        return root;
    }


    public TreeNode sortedListToBSTV2(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode middleNode = getMiddleNode(head);
        TreeNode root = new TreeNode(middleNode.val);
        if (middleNode == head) {
            return root;
        }
        root.left = sortedListToBSTV2(head);
        root.right = sortedListToBSTV2(middleNode.next);
        return root;
    }

    private ListNode getMiddleNode(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode prev = null;
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        if (prev != null) {
            prev.next = null;
        }
        return slow;
    }


    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        if (Math.abs(left - right) > 1) {
            return false;
        }
        return isBalanced(root.left) && isBalanced(root.right);
    }


}
