package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.Node;
import org.code.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author dora
 * @date 2020/8/10
 */
public class TwoPage {

    private int maxPathSumValue = 0;

    public static void main(String[] args) {
        TwoPage page = new TwoPage();

//        Node root = new Node(1);
//        Node left1 = new Node(2);
//        Node right1 = new Node(3);
//        root.left = left1;
//        root.right = right1;


        page.getRow(3);
    }

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

    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        if (root.val == sum && root.left == null && root.right == null) {
            return true;
        }
        return hasPathSum(root.left, sum - root.val) | hasPathSum(root.right, sum - root.val);
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalPathSum(result, new ArrayList<>(), root, sum);

        return result;
    }

    private void intervalPathSum(List<List<Integer>> result, ArrayList<Integer> tmp, TreeNode root, int sum) {
        tmp.add(root.val);
        if (root.val == sum && root.left == null && root.right == null) {
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

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        stack.push(p);
        while (!stack.isEmpty()) {
            p = stack.pop();
            if (prev != null) {
                prev.right = p;
                prev.left = null;
            }
            if (p.right != null) {
                stack.push(p.right);
            }
            if (p.left != null) {
                stack.push(p.left);
            }
            prev = p;
        }
    }

    public int numDistinct(String s, String t) {
        if (s == null || t == null) {
            return 0;
        }
        int m = s.length();
        int n = t.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = dp[i - 1][j] + (s.charAt(i - 1) == t.charAt(j - 1) ? dp[i - 1][j - 1] : 0);
            }
        }
        return dp[m][n];
    }

    /**
     * 116. 填充每个节点的下一个右侧节点指针
     * todo
     *
     * @param root
     * @return
     */
    public Node connect(Node root) {
        if (root == null || root.next == null) {
            return root;
        }
        Node currentNode = root;
        while (currentNode.left != null) {
            Node nextLevel = currentNode.left;
            while (currentNode.right != null) {
                nextLevel.next = currentNode.right;
                if (currentNode.next != null) {
                    currentNode.right.next = currentNode.next.left;
                }
                currentNode = currentNode.next;
            }
            currentNode = nextLevel;
        }
        return root;

    }

    /**
     * 117. 填充每个节点的下一个右侧节点指针 II
     * todo
     *
     * @param root
     * @return
     */
    public Node connectV2(Node root) {
        return null;
    }

    public List<List<Integer>> generate(int numRows) {
        if (numRows < 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();


        for (int i = 0; i < numRows; i++) {

            List<Integer> tmp = new ArrayList<>();

            tmp.add(1);

            for (int j = i - 1; j >= 1; j--) {
                int value = result.get(i - 1).get(j) + result.get(i - 1).get(j - 1);
                tmp.add(value);
            }
            if (i > 0) {
                tmp.add(1);
            }
            result.add(tmp);
        }
        return result;
    }

    public List<Integer> getRow(int rowIndex) {
        if (rowIndex < 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>(rowIndex);

        result.add(1);

        for (int i = 1; i <= rowIndex; i++) {

            for (int j = i - 1; j >= 1; j--) {
                int value = result.get(j - 1) + result.get(j);

                result.set(j, value);
            }
            if (i > 0) {
                result.add(1);
            }
        }
        return result;
    }

    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty()) {
            return Integer.MAX_VALUE;
        }
        int size = triangle.size();
        List<Integer> result = triangle.get(size - 1);

        for (int i = size - 2; i >= 0; i--) {
            List<Integer> currentRow = triangle.get(i);

            int currentSize = currentRow.size();

            for (int j = 0; j < currentSize; j++) {
                int value = currentRow.get(j) + Math.min(result.get(j), result.get(j + 1));

                result.set(j, value);
            }
        }
        return result.get(0);
    }

    public int maxProfit(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            }
            result = Math.max(result, prices[i] - minPrice);
        }
        return result;
    }

    public int maxProfitII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int result = 0;
        int minPrice = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > minPrice) {
                result += prices[i] - minPrice;
            }
            minPrice = prices[i];
        }
        return result;
    }

    /**
     * todo 卖股票最佳利润
     *
     * @param prices
     * @return
     */
    public int maxProfitIII(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int minLeftPrice = prices[0];
        int leftResult = 0;
        int[] leftProfit = new int[prices.length];

        for (int i = 1; i < prices.length; i++) {
            if (minLeftPrice > prices[i]) {
                minLeftPrice = prices[i];
            }
            leftResult = Math.max(leftResult, prices[i] - minLeftPrice);

            leftProfit[i] = leftResult;
        }

        int rightResult = 0;

        int minRightPrice = prices[prices.length - 1];

        int[] rightProfit = new int[prices.length + 1];

        for (int i = prices.length - 2; i >= 0; i--) {
            if (minRightPrice < prices[i]) {
                minRightPrice = prices[i];
            }
            rightResult = Math.max(rightResult, minRightPrice - prices[i]);

            rightProfit[i] = rightResult;
        }
        int result = 0;
        for (int i = 0; i < prices.length; i++) {
            result = Math.max(result, leftProfit[i] + rightProfit[i + 1]);
        }
        return result;
    }

    public int maxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        intervalMaxPathSum(root);
        return maxPathSumValue;
    }

    private int intervalMaxPathSum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int leftValue = intervalMaxPathSum(root.left);
        int rightValue = intervalMaxPathSum(root.right);

        leftValue = Math.max(leftValue, 0);
        rightValue = Math.max(rightValue, 0);

        maxPathSumValue = Math.max(maxPathSumValue, root.val + leftValue + rightValue);
        return root.val + Math.max(leftValue, rightValue);

    }


    public boolean isPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            while (start < end && !Character.isLetterOrDigit(s.charAt(start))) {
                start++;
            }
            while (start < end && !Character.isLetterOrDigit(s.charAt(end))) {
                end--;
            }
            if (start < end) {
                if (Character.toLowerCase(s.charAt(start)) != Character.toLowerCase(s.charAt(end))) {
                    return false;
                }
                start++;
                end--;
            }
        }
        return true;
    }


    /**
     * 126. 单词接龙 II
     * todo
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        return null;
    }


    /**
     * 127. 单词接龙
     * todo
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        return 0;
    }


    /**
     * 128. 最长连续序列
     *
     * @param nums
     * @return
     */
    public int longestConsecutive(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        HashMap<Integer, Integer> map = new HashMap<>();

        int result = 0;

        for (int num : nums) {
            if (map.containsKey(num)) {
                continue;
            }
            Integer leftSide = map.getOrDefault(num - 1, 0);
            Integer rightSide = map.getOrDefault(num + 1, 0);

            int value = leftSide + rightSide + 1;

            result = Math.max(result, value);

            map.put(num - leftSide, value);
            map.put(num + rightSide, value);
            map.put(num, value);
        }
        return result;
    }

    public int sumNumbers(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return intervalSumNumbers(0, root);
    }

    private int intervalSumNumbers(int val, TreeNode root) {
        if (root == null) {
            return 0;
        }
        int tmp = val * 10 + root.val;
        if (root.left == null && root.right == null) {
            return tmp;
        }
        return intervalSumNumbers(tmp, root.left) + intervalSumNumbers(tmp, root.right);
    }


}
