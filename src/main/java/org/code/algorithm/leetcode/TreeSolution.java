package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.Node;
import org.code.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2020/10/6
 */
public class TreeSolution {

    private int maxConsecutiveResult = 0;


    //---普通题- //

    public static void main(String[] args) {
        TreeSolution solution = new TreeSolution();
        int[][] matrix = new int[][]{{2147483647, -1, 0, 2147483647},
                {2147483647, 2147483647, 2147483647, -1}, {2147483647, -1, 2147483647, -1}, {0, -1, 2147483647, 2147483647}};
        int[] preorder = new int[]{1, 3, 2};
        TreeNode root = new TreeNode(2);
        root.left = new TreeNode(1);
        double tmp = 2147483647.0;
        solution.closestValue(root, tmp);
    }

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


    //--二叉树的遍历问题-- //
    //--- traversal --- //

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
        LinkedList<Integer> result = new LinkedList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (!stack.isEmpty() || p != null) {
            if (p != null) {
                result.addFirst(p.val);
                stack.push(p);
                p = p.right;
            } else {
                p = stack.pop();
                p = p.left;
            }
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

    /**
     * 103. Binary Tree Zigzag Level Order Traversal
     *
     * @param root
     * @return
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        List<List<Integer>> result = new ArrayList<>();
        queue.offer(root);
        boolean leftToRight = true;
        while (!queue.isEmpty()) {
            int size = queue.size();
            LinkedList<Integer> tmp = new LinkedList<>();
            for (int i = 0; i < size; i++) {
                TreeNode poll = queue.poll();

                if (poll.left != null) {
                    queue.offer(poll.left);
                }
                if (poll.right != null) {
                    queue.offer(poll.right);
                }
                if (leftToRight) {
                    tmp.addLast(poll.val);
                } else {
                    tmp.addFirst(poll.val);
                }
            }
            leftToRight = !leftToRight;
            result.add(tmp);
        }
        return result;
    }

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


    // --二叉树深度问题- //

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

    //--路径问题--//

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


    // --构造二叉树-- //

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


    //----二叉搜索树-------//

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
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != tail && fast.next.next != tail) {
            fast = fast.next.next;
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
        ListNode slow = head;

        ListNode fast = head;

        ListNode prev = slow;

        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            prev = slow;
            slow = slow.next;
        }
        TreeNode root = new TreeNode(slow.val);

        ListNode next = slow.next;

        prev.next = null;

        root.left = sortedListToBST(head);

        root.right = sortedListToBST(next);

        return root;
    }

    // ----填充下一列--- //
    // ---要求使用常量内存空间---//

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode prev = null;
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
            if (pop.right != null) {
                stack.push(pop.right);
            }
            if (pop.left != null) {
                stack.push(pop.left);
            }
            if (prev != null) {
                prev.right = pop;
                prev.left = null;
            }
            prev = pop;
        }
    }

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
        Node traversal = root;
        while (traversal.left != null) {
            Node current = traversal;

            Node next = traversal.left;

            while (current != null) {
                current.left.next = current.right;
                if (current.next != null) {
                    current.right.next = current.next.left;
                }
                current = current.next;
            }
            traversal = next;
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
        Node traversal = root;

        Node level = null;

        Node prev = null;

        while (traversal != null) {
            while (traversal != null) {
                if (traversal.left != null) {
                    if (level == null) {
                        level = traversal.left;
                    } else {
                        prev.next = traversal.left;
                    }
                    prev = traversal.left;
                }
                if (traversal.right != null) {
                    if (level == null) {
                        level = traversal.right;
                    } else {
                        prev.next = traversal.right;
                    }
                    prev = traversal.right;
                }
                traversal = traversal.next;
            }
            traversal = level;

            level = null;

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
        TreeNode left = root.left;

        TreeNode node = upsideDownBinaryTree(left);

        left.left = root.right;

        left.right = root;

        root.left = null;

        root.right = null;

        return node;
    }

    public TreeNode upsideDownBinaryTreeV2(TreeNode root) {
        if (root == null || root.left == null) {
            return root;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (p.left != null) {
            stack.push(p.left);
            p = p.left;
        }
        TreeNode newHead = stack.peek();
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
            if (!stack.isEmpty()) {
                TreeNode peek = stack.peek();
                pop.left = peek.right;
                pop.right = peek;
            } else {
                pop.left = root.right;
                pop.right = root;
                root.left = null;
                root.right = null;
            }
        }
        return newHead;
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

    /**
     * todo
     * 255 Verify Preorder Sequence in Binary Search Tree
     *
     * @param preorder: List[int]
     * @return: return a boolean
     */
    public boolean verifyPreorder(int[] preorder) {
        // write your code here
        if (preorder == null || preorder.length == 0) {
            return false;
        }
        return intervalVerifyPreorder(Integer.MIN_VALUE, 0, preorder.length - 1, preorder, Integer.MAX_VALUE);
    }

    private boolean intervalVerifyPreorder(int minValue, int start, int end, int[] preorder, int maxValue) {
        if (start > end) {
            return true;
        }
        int current = preorder[start];
        if (current <= minValue || current >= maxValue) {
            return false;
        }
        int index = 0;
        for (int i = start + 1; i <= end; i++) {
            if (preorder[i] >= current) {
                index = i;
                break;
            }
        }
        return intervalVerifyPreorder(minValue, start + 1, index - 1, preorder, current)
                && intervalVerifyPreorder(current, index, end, preorder, maxValue);
    }


    public boolean verifyPreorderV2(int[] preorder) {
        // write your code here

        if (preorder == null || preorder.length == 0) {
            return false;
        }
        Stack<Integer> stack = new Stack<>();
        int low = Integer.MIN_VALUE;
        for (int val : preorder) {

            if (val < low) {
                return false;
            }
            while (!stack.isEmpty() && val > stack.peek()) {
                low = stack.pop();
            }
            stack.push(val);
        }
        return true;
    }

    /**
     * 257
     * Binary Tree Paths
     *
     * @param root: the root of the binary tree
     * @return: all root-to-leaf paths
     */
    public List<String> binaryTreePaths(TreeNode root) {
        // write your code here
        if (root == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalBinaryTreePaths(result, root, "");
        return result;
    }

    private void intervalBinaryTreePaths(List<String> result, TreeNode root, String s) {
        if (root == null) {
            return;
        }
        if (s.isEmpty()) {
            s = s + root.val;
        } else {
            s = s + "->" + root.val;
        }
        if (root.left == null && root.right == null) {
            result.add(s);
            return;
        }

        intervalBinaryTreePaths(result, root.left, s);
        intervalBinaryTreePaths(result, root.right, s);

    }

    /**
     * @param root:   the given BST
     * @param target: the given target
     * @param k:      the given k
     * @return: k values in the BST that are closest to the target
     */
    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        // write your code here
        if (root == null) {
            return new ArrayList<>();
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(k, (o1, o2) -> o2 - o1);
        TreeNode p = root;
        while (p != null) {
            int size = priorityQueue.size();
            if (size < k) {
//                priorityQueue.offer()
            }
        }
        return null;
    }

    /**
     * 285 Inorder Successor in BST
     *
     * @param root: The root of the BST.
     * @param p:    You need find the successor node of p.
     * @return: Successor of p.
     */
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (root == null || p == null) {
            return null;
        }
        // write your code here
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || current != null) {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            current = stack.pop();

            if (prev != null && prev == p) {
                return current;
            }
            prev = current;
            current = current.right;
        }
        return null;
    }

    // ---二叉树连续序列问题---//

    /**
     * 298 Binary Tree Longest Consecutive Sequence
     *
     * @param root: the root of binary tree
     * @return: the length of the longest consecutive sequence path
     */
    public int longestConsecutive2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        intervalLongestConsecutive2(root, root.val, 0);
        return maxConsecutiveResult;
        // write your code here
    }

    private int intervalLongestConsecutive2(TreeNode root, int val, int result) {
        if (root == null) {
            return 0;
        }
        if (root.val == val + 1) {
            result++;
        } else {
            result = 1;
        }
        maxConsecutiveResult = Math.max(result, maxConsecutiveResult);
        intervalLongestConsecutive2(root.left, root.val, result);
        intervalLongestConsecutive2(root.right, root.val, result);
        return result;
    }


    /**
     * todo
     * 314 Binary Tree Vertical Order Traversal
     *
     * @param root: the root of tree
     * @return: the vertical order traversal
     */
    public List<List<Integer>> verticalOrder(TreeNode root) {
        // write your code here
        return null;
    }

    /**
     * 270
     * Closest Binary Search Tree Value
     *
     * @param root:   the given BST
     * @param target: the given target
     * @return: the value in the BST that is closest to the target
     */
    public int closestValue(TreeNode root, double target) {
        // write your code here
        double result = 0;
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (prev == null) {
                result = p.val;
            } else {
                result = Math.abs(result - target) - Math.abs(p.val - target) < 0 ? result : p.val;
            }
            prev = p;
            p = p.right;
        }
        return (int) result;
    }


    public int closestValueV2(TreeNode root, double target) {
        double result = 0;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode pop = stack.pop();
            if (pop.val == target) {
                return (int) target;
            }
            double diff = Math.abs(pop.val - target);
            double prev = Math.abs(result - target);
            if (diff < prev) {
                result = pop.val;
                if (pop.right != null) {
                    stack.push(pop.right);
                }
            } else {
                if (pop.left != null) {
                    stack.push(pop.left);
                }
            }
        }
        return (int) result;
    }


    /**
     * todo
     * 272 Closest Binary Search Tree Value II
     *
     * @param root:   the given BST
     * @param target: the given target
     * @param k:      the given k
     * @return: k values in the BST that are closest to the target
     */
    public List<Integer> closestKValuesII(TreeNode root, double target, int k) {
        // write your code here
        if (root == null) {
            return new ArrayList<>();
        }
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(k, Comparator.reverseOrder());
        return null;
    }


}
