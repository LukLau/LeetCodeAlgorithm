package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.Node;
import org.code.algorithm.datastructe.TreeNode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

/**
 * @author dora
 * @date 2020/10/6
 */
public class TreeSolution {

    public static void main(String[] args) {
        TreeSolution solution = new TreeSolution();
    }


    //---普通题- //

    /**
     * 129. Sum Root to Leaf Numbers
     *
     * @param root
     * @return
     */
    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return intervalsSumNumbers(root, 0);
    }

    private int intervalsSumNumbers(TreeNode root, int val) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return val * 10 + root.val;
        }
        return intervalsSumNumbers(root.left, val * 10 + root.val)
                + intervalsSumNumbers(root.right, val * 10 + root.val);
    }


    //--二叉树的遍历问题-- //

    /**
     * 94. Binary Tree Inorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            result.add(p.val);
            p = p.right;
        }
        return result;
    }

    /**
     * @param root
     * @return
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<Integer> result = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            if (p != null) {
                stack.push(p);
                result.addFirst(p.val);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
        }
        return result;
    }

    // --生成二叉搜索树- //

    public List<TreeNode> generateTrees(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        return intervalGenerateTrees(1, n);

    }

    private List<TreeNode> intervalGenerateTrees(int start, int end) {
        List<TreeNode> result = new ArrayList<>();
        if (start > end) {
            result.add(null);
            return result;
        }
        if (start == end) {
            TreeNode node = new TreeNode(start);
            result.add(node);
            return result;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftNodes = intervalGenerateTrees(start, i - 1);
            List<TreeNode> rightNodes = intervalGenerateTrees(i + 1, end);
            for (TreeNode leftNode : leftNodes) {
                for (TreeNode rightNode : rightNodes) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftNode;
                    root.right = rightNode;
                    result.add(root);
                }
            }
        }
        return result;
    }


    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 0) {
            return 0;
        }
        if (n <= 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }


    // --二叉树深度问题- //


    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
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


    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        if (root.left == null) {
            return 1 + minDepth(root.right);
        }
        if (root.right == null) {
            return 1 + minDepth(root.left);
        }
        return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }

    //--路径问题--//


    /**
     * 112. Path Sum
     *
     * @param root
     * @param sum
     * @return
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.left == null && root.right == null && root.val == sum) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) || hasPathSum(root.right, sum - root.val);
    }


    /**
     * 113. Path Sum II
     *
     * @param root
     * @param sum
     * @return
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalPathSum(result, new ArrayList<Integer>(), root, sum);
        return result;
    }

    private void intervalPathSum(List<List<Integer>> result, List<Integer> tmp, TreeNode root, int sum) {
        tmp.add(root.val);
        if (root.left == null && root.right == null && root.val == sum) {
            result.add(new ArrayList<>(tmp));
        } else {
            if (root.left != null) {
                intervalPathSum(result, tmp, root.left, sum - root.val);
            }
            if (root.right != null) {
                intervalPathSum(result, tmp, root.right, sum - root.val);
            }
        }
        tmp.remove(tmp.size() - 1);
    }


    // --构造二叉树-- //

    /**
     * 105. Construct Binary Tree from Preorder and Inorder Traversal
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null) {
            return null;
        }
        return intervalBuildTree(0, preorder, 0, inorder.length - 1, inorder);
    }

    private TreeNode intervalBuildTree(int preStart, int[] preorder, int inStart, int inEnd, int[] inorder) {
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
        root.left = intervalBuildTree(preStart + 1, preorder, inStart, index - 1, inorder);

        root.right = intervalBuildTree(preStart + index - inStart + 1, preorder, index + 1, inEnd, inorder);

        return root;
    }

    public TreeNode buildTreeV2(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null) {
            return null;
        }
        return intervalBuildTreeV2(0, inorder.length - 1, inorder, 0, postorder.length - 1, postorder);

    }

    private TreeNode intervalBuildTreeV2(int inStart, int inEnd, int[] inorder, int postStart, int postEnd, int[] postorder) {
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
        root.left = intervalBuildTreeV2(inStart, index - 1, inorder, postStart, postStart + index - inStart - 1, postorder);

        root.right = intervalBuildTreeV2(index + 1, inEnd, inorder, postStart + index - inStart, postEnd - 1, postorder);
        return root;
    }


    //----二叉搜索树-------//

    /**
     * 108. Convert Sorted Array to Binary Search Tree
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        return intervalSortedArrayToBST(nums, 0, nums.length - 1);
    }

    private TreeNode intervalSortedArrayToBST(int[] nums, int start, int end) {
        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = intervalSortedArrayToBST(nums, start, mid - 1);
        root.right = intervalSortedArrayToBST(nums, mid + 1, end);
        return root;
    }


    /**
     * 109. Convert Sorted List to Binary Search Tree
     *
     * @param head
     * @return
     */
    public TreeNode sortedListToBSTV2(ListNode head) {
        if (head == null) {
            return null;
        }
        return intervalSortedListToBST(head, null);
    }

    private TreeNode intervalSortedListToBST(ListNode head, ListNode tail) {
        if (head == tail) {
            return null;
        }
        ListNode fastNode = head;
        ListNode slow = fastNode;
        while (fastNode != tail && fastNode.next != tail) {
            fastNode = fastNode.next.next;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);

        root.left = intervalSortedListToBST(head, slow);

        root.right = intervalSortedListToBST(slow.next, tail);

        return root;
    }


    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next == null) {
            return new TreeNode(head.val);
        }
        ListNode fast = head;
        ListNode slow = head;
        ListNode prev = slow;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = slow;
            slow = slow.next;
        }
        prev.next = null;
        TreeNode root = new TreeNode(slow.val);
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);
        return root;
    }


    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        TreeNode prev = null;
        while (!stack.isEmpty()) {

            TreeNode pop = stack.pop();
            if (pop.right != null) {
                stack.push(pop.right);
            }
            if (pop.left != null) {
                stack.push(pop.left);
            }
            if (prev != null) {
                prev.left = null;
                prev.right = pop;
            }
            prev = pop;
        }
    }


    // ----填充下一列--- //
    // ---要求使用常量内存空间---//

    /**
     * 116. Populating Next Right Pointers in Each Node
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        if (root.left == null) {
            return root;
        }
        Node currentNode = root;

        while (currentNode.left != null) {
            Node levelRoot = currentNode.left;

            while (currentNode != null) {
                Node leftNode = currentNode.left;

                leftNode.next = currentNode.right;

                if (currentNode.next != null) {

                    leftNode.next.next = currentNode.next.left;
                }
                currentNode = currentNode.next;
            }
            currentNode = levelRoot;
        }
        return root;
    }


    /**
     * todo
     * 117. Populating Next Right Pointers in Each Node II
     *
     * @param root
     * @return
     */
    public Node connectV2(Node root) {
        if (root == null) {
            return null;
        }
        Node currentNode = root;
        Node levelRoot = null;
        Node prev = null;
        while (currentNode != null) {

            while (currentNode != null) {
                if (currentNode.left != null) {
                    if (levelRoot == null) {
                        levelRoot = currentNode.left;
                    } else {
                        prev.next = currentNode.left;
                    }
                    prev = currentNode.left;
                }
                if (currentNode.right != null) {
                    if (levelRoot == null) {
                        levelRoot = currentNode.right;
                    } else {
                        prev.next = currentNode.right;
                    }
                    prev = currentNode.right;
                }
                currentNode = currentNode.next;
            }
            currentNode = levelRoot;
            levelRoot = null;
            prev = null;
        }
        return root;

    }


    /**
     * 156 Binary Tree Upside Down
     * Medium
     *
     * @param root: the root of binary tree
     * @return: new root
     */
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        // write your code here
        if (root == null || root.left == null) {
            return root;
        }
        TreeNode leftNode = root.left;

        TreeNode newRoot = upsideDownBinaryTree(root.left);

        leftNode.left = root.right;

        leftNode.right = root;

        root.left = null;

        root.right = null;

        return newRoot;
    }

    public TreeNode upsideDownBinaryTreeV2(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (p != null) {
            stack.push(p);
            p = p.left;
        }
        TreeNode newRoot = stack.peek();

        while (!stack.isEmpty()) {
            TreeNode popNode = stack.pop();

            TreeNode peekNode = stack.isEmpty() ? null : stack.peek();

            if (peekNode != null) {
                popNode.left = peekNode.right;

                popNode.right = peekNode;

                peekNode.left = null;

                peekNode.right = null;
            }
        }
        return newRoot;
    }


    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode tmp = root.left;

        root.left = root.right;

        root.right = tmp;

        invertTree(root.left);

        invertTree(root.right);

        return root;
    }


    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;

        int iteratorCount = 0;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            iteratorCount++;
            if (iteratorCount == k) {
                return p.val;
            }
            p = p.right;
        }
        return -1;
    }


}
