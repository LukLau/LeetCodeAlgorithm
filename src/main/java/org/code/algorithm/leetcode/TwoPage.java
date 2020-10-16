package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.Node;
import org.code.algorithm.datastructe.Point;
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


//        String s = "catsanddog";
//
//        ListNode head = new ListNode(1);
//
//        ListNode two = new ListNode(2);
//
//        head.next = two;
//
//        ListNode three = new ListNode(3);
//
//        two.next = three;
//
//        three.next = new ListNode(4);
//
//        three.next.next = new ListNode(5);

//        List<String> wordDict = Arrays.asList("cat", "cats", "and", "sand", "dog");
//        page.wordBreakV2(s, wordDict);

//        page.reorderList(head);
//        String word = "the sky is blue";
//        page.reverseWords(word.toCharArray());

        int[] nums = new int[]{1, 2, 3, 1};
        page.rob(nums);
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
        if (s == null) {
            return false;
        }
        s = s.trim();
        int begin = 0;
        int end = s.length() - 1;
        while (begin < end) {
            while (begin < end && !Character.isLetterOrDigit(s.charAt(begin))) {
                begin++;
            }
            while (begin < end && !Character.isLetterOrDigit(s.charAt(end))) {
                end--;
            }
            if (Character.toLowerCase(s.charAt(begin)) != Character.toLowerCase(s.charAt(end))) {
                return false;
            } else {
                begin++;
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

            int tmp = leftSide + rightSide + 1;

            result = Math.max(result, tmp);

            map.put(num - leftSide, tmp);

            map.put(num + rightSide, tmp);

            map.put(num, tmp);
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


    /**
     * 130. 被围绕的区域
     *
     * @param board
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];


        // 深度优先遍历
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {

                // 只遍历边界情况
                if (checkEdge(i, j, board) && board[i][j] == 'O') {
                    intervalSolve(used, i, j, board);
                }
            }
        }
        // 对于非边界连续结点设置相反的符号
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = '-';
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == '-') {
                    board[i][j] = 'X';
                } else if (board[i][j] == '+') {
                    board[i][j] = 'O';
                }
            }
        }

    }

    private boolean checkEdge(int i, int j, char[][] board) {
        if (i == 0 || i == board.length - 1) {
            return true;
        }
        return j == 0 || j == board[i].length - 1;
    }

    private void intervalSolve(boolean[][] used, int i, int j, char[][] board) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length) {
            return;
        }
        // 已经遍历过直接跳过去
        if (used[i][j]) {
            return;
        }
        used[i][j] = true;
        if (board[i][j] == 'X') {
            return;
        }
        board[i][j] = '+';
        intervalSolve(used, i - 1, j, board);
        intervalSolve(used, i + 1, j, board);
        intervalSolve(used, i, j - 1, board);
        intervalSolve(used, i, j + 1, board);
    }


    public void solveV2(char[][] board) {

    }


    /**
     * todo
     * 131. 分割回文串
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<List<String>> result = new ArrayList<>();

        int index = 1;
        while (index < s.length()) {
            String tmp = s.substring(0, index);

//            if (checkPalindrome(tmp) && )

        }
        return null;
    }

    private boolean checkPalindrome(String s) {
        if (s.isEmpty()) {
            return true;
        }
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            if (s.charAt(start) != s.charAt(end)) {
                return false;
            }
            start++;
            end--;

        }
        return true;

    }


    public int minCut(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int len = s.length();
        int[] cut = new int[len];
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            int minCut = i;
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i) && (i - j < 2 || dp[j + 1][i - 1])) {
                    dp[j][i] = true;

                    minCut = j == 0 ? 0 : Math.min(minCut, cut[j - 1] + 1);
                }
            }
            cut[i] = minCut;
        }
        return cut[len - 1];
    }

    /**
     * todo 图的算法
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        return null;
    }


    /**
     * 134. 加油站
     *
     * @param gas
     * @param cost
     * @return
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        if (gas == null || gas.length == 0 || cost == null || cost.length == 0) {
            return -1;
        }
        int globalCost = 0;
        int localCost = 0;
        int index = -1;
        for (int i = 0; i < gas.length; i++) {
            globalCost += gas[i] - cost[i];

            localCost += gas[i] - cost[i];

            if (localCost < 0) {
                index = i;
                localCost = 0;
            }
        }

        return globalCost < 0 ? -1 : index + 1;
    }

    public int candy(int[] ratings) {
        if (ratings == null || ratings.length == 0) {
            return 0;
        }
        int[] dp = new int[ratings.length];
        Arrays.fill(dp, 1);
        int result = 0;
        for (int j = 1; j <= ratings.length; j++) {
            if (ratings[j - 1] < ratings[j] && dp[j] < dp[j - 1] + 1) {
                dp[j] = dp[j - 1] + 1;
            }
        }
        for (int j = ratings.length - 2; j >= 0; j--) {
            if (ratings[j] > ratings[j + 1] && dp[j] < dp[j + 1] + 1) {
                dp[j] = dp[j + 1] + 1;
            }

        }
        for (int candyNum : dp) {
            result += candyNum;
        }
        return result;
    }


    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {
            result ^= num;
        }
        return result;
    }


    /**
     * todo 137. 只出现一次的数字 II
     *
     * @param nums
     * @return
     */
    public int singleNumberV2(int[] nums) {

        return -1;
    }


    public Node copyRandomList(Node head) {
        if (head == null) {
            return null;
        }

        Node currentNode = head;
        while (currentNode != null) {
            Node tmp = new Node(currentNode.val);
            if (currentNode.next != null) {
                tmp.next = currentNode.next;
            }
            currentNode.next = tmp;
            currentNode = tmp.next;
        }
        currentNode = head;
        while (currentNode != null) {
            Node tmp = currentNode.next;
            if (currentNode.random != null) {
                tmp.random = currentNode.random.next;
            }
            currentNode = tmp.next;
        }
        currentNode = head;

        Node randomHead = head.next;
        while (currentNode.next != null) {
            Node tmp = currentNode.next;
            currentNode.next = tmp.next;
            currentNode = tmp;
        }
        return randomHead;
    }


    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        HashMap<String, Boolean> map = new HashMap<>();
        return intervalWordBreak(map, s, wordDict);
    }

    private boolean intervalWordBreak(HashMap<String, Boolean> map, String s, List<String> wordDict) {
        if (s.isEmpty()) {
            return true;
        }
        if (map.containsKey(s)) {
            return map.get(s);
        }
        for (String dict : wordDict) {
            int index = s.indexOf(dict);
            if (index != 0) {
                continue;
            }
            String substring = s.substring(dict.length());
            if (intervalWordBreak(map, substring, wordDict)) {
                map.put(s, true);
                return true;
            }
        }
        map.put(s, false);
        return false;
    }


    public List<String> wordBreakV2(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        return intervalWordBreakV2(map, s, wordDict);
    }

    private List<String> intervalWordBreakV2(HashMap<String, List<String>> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        if (s.isEmpty()) {
            return Arrays.asList("");
        }
        List<String> result = new ArrayList<>();
        for (String word : wordDict) {
            int index = s.indexOf(word);
            if (index != 0) {
                continue;
            }
            List<String> tmp = intervalWordBreakV2(map, s.substring(word.length()), wordDict);
            for (String t : tmp) {
                result.add(word + (t.isEmpty() ? "" : " ") + t);
            }
        }
        map.put(s, result);
        return result;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    /**
     * todo
     * 143. 重排链表
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode slow = head;
        ListNode fast = head;

        // 遍历链表 取中间结点
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        // 中间结点之后的结点进行反转
        ListNode reverseNode = reverseListNode(slow.next);

        // 断开成两个链表
        slow.next = null;

        slow = head;

        // 重新排序 组装
        while (reverseNode != null) {
            ListNode tmp = reverseNode.next;

            reverseNode.next = slow.next;

            slow.next = reverseNode;

            slow = reverseNode.next;

            reverseNode = tmp;

        }
    }

    private ListNode reverseListNode(ListNode fast) {
        ListNode prev = null;
        while (fast != null) {
            ListNode tmp = fast.next;
            fast.next = prev;
            prev = fast;
            fast = tmp;
        }
        return prev;
    }


    /**
     * 145. 二叉树的后序遍历
     * todo
     *
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
     * todo 链表的插入排序
     *
     * @param head
     * @return
     */
    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        return null;
    }

    /**
     * todo 排序链表 O(nlogN)
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        return null;
    }


    /**
     * todo
     * 149. 直线上最多的点数
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }
        if (points.length <= 2) {
            return points.length;
        }
        // 封装成坐标轴点
        Point[] array = new Point[points.length];
        for (int i = 0; i < points.length; i++) {
            array[i] = new Point(points[i][0], points[i][1]);
        }
        int result = 0;

        for (int i = 0; i < array.length; i++) {

            Map<Integer, Map<Integer, Integer>> map = new HashMap<>();

            int overlap = 0;

            int max = 0;
            for (int j = i + 1; j < array.length; j++) {
                int x = array[j].x - array[i].x;
                int y = array[j].y - array[i].y;

                if (x == 0 && j == 0) {
                    if (x == 0 && y == 0) {
                        overlap++;
                        continue;
                    }
                    int gcd = generateGCD(x, y);
                    if (gcd != 0) {
                        x /= gcd;

                        y /= gcd;
                    }
                    if (map.containsKey(x)) {
                        if (map.get(x).containsKey(y)) {
                            map.get(x).put(y, map.get(x).get(y) + 1);
                        } else {
                            map.get(x).put(y, 1);
                        }
                    } else {
                        Map<Integer, Integer> m = new HashMap<>();
                        m.put(y, 1);
                        map.put(x, m);
                    }
                    max = Math.max(max, map.get(x).get(y));
                }
                result = Math.max(result, max + overlap + 1);
            }
        }
        return result;
    }


    /**
     * 使用辗转相除法
     * a % b = m
     * b % m = n
     * ....
     *
     * @param x
     * @param y
     * @return
     */
    private int generateGCD(int x, int y) {
        if (y == 0) {
            return x;
        }
        return generateGCD(y, x % y);
    }


    public int maxPointsV2(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }
        if (points.length <= 2) {
            return points.length;
        }
        int result = 0;


        for (int i = 0; i < points.length; i++) {

            int max = 0;

            int overlap = 0;


            HashMap<String, Integer> map = new HashMap<>();

            for (int j = i + 1; j < points.length; j++) {
                int x = points[j][0] - points[i][0];
                int y = points[j][1] - points[i][1];

                if (x == 0 && y == 0) {
                    overlap++;
                    continue;
                }
                int gcd = generateGCD(x, y);

                x /= gcd;

                y /= gcd;

                String key = x + "@" + y;

                Integer value = map.getOrDefault(key, 0);

                value = value + 1;

                map.put(key, value);

                max = Math.max(max, value);
            }
            result = Math.max(result, max + overlap + 1);
        }
        return result;

    }


    /**
     * 逆波兰
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < tokens.length; i++) {
            String token = tokens[i];
        }
        return -1;
    }

    public String reverseWords(String s) {
        if (s == null) {
            return "";
        }
        s = s.trim();

        String[] words = s.split(" ");
        StringBuilder builder = new StringBuilder();
        for (int i = words.length - 1; i >= 0; i--) {
            String word = words[i];
            if ("".equals(word)) {
                continue;
            }
            builder.append(word);
            if (i > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = nums[0];
        int minValue = nums[0];
        int maxValue = nums[0];

        for (int i = 1; i < nums.length; i++) {
            int tmpMax = Math.max(Math.max(maxValue * nums[i], minValue * nums[i]), nums[i]);

            int tmpMin = Math.min(Math.min(maxValue * nums[i], minValue * nums[i]), nums[i]);

            result = Math.max(result, tmpMax);

            maxValue = tmpMax;

            minValue = tmpMin;
        }
        return result;

    }


    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        int left = 0;

        int right = nums.length - 1;
        while (left < right) {
            if (nums[left] < nums[right]) {
                return nums[left];
            }
            int mid = left + (right - left) / 2;


            if (nums[left] <= nums[mid]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return nums[left];
    }


    public int findMinV2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] <= nums[right]) {
                right = mid;
            } else {
                left = mid + 1;

            }
        }
        return nums[left];
    }


    public int findMinInRepeatNums(int[] nums) {
        if (nums == null || nums.length == 0) {
            return Integer.MAX_VALUE;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == nums[right]) {
                right--;
            } else if (nums[mid] < nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];

    }


    public int findMinInRepeatNumsV2(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            if (nums[left] < nums[right]) {
                return nums[left];
            }
            int mid = left + (right - left) / 2;

            if (nums[left] < nums[mid]) {
                left = mid + 1;
            } else if (nums[left] > nums[mid]) {
                right = mid;
            } else {
                left++;
            }
        }
        return nums[left];

    }


    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA;

        ListNode p2 = headB;
        while (p1 != p2) {
            p1 = p1 == null ? headB : p1.next;
            p2 = p2 == null ? headA : p2.next;
        }
        return p1;

    }


    public int hammingWeight(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }
        return count;

    }


    public int trailingZeroes(int n) {
        int count = 0;
        while (n / 5 != 0) {
            count += n / 5;
            n /= 5;
        }
        return count;
    }


    public int calculateMinimumHP(int[][] dungeon) {
        int row = dungeon.length;

        int column = dungeon[0].length;

        int[][] dp = new int[row][column];

        for (int i = row - 1; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                if (i == row - 1 && j == column - 1) {
                    dp[i][j] = Math.max(1, 1 - dungeon[i][j]);
                } else if (i == row - 1) {
                    dp[i][j] = Math.max(1, dp[i][j + 1] - dungeon[i][j]);
                } else if (j == column - 1) {
                    dp[i][j] = Math.max(1, dp[i + 1][j] - dungeon[i][j]);
                } else {
                    dp[i][j] = Math.max(1, -dungeon[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]));
                }
            }
        }
        return dp[0][0];
    }

    public int calculateMinimumHPV2(int[][] dungeon) {
        int row = dungeon.length;
        int column = dungeon[0].length;
        int[] dp = new int[column];

        for (int j = column - 1; j >= 0; j--) {
            if (j == column - 1) {
                dp[j] = Math.max(1, 1 - dungeon[row - 1][j]);
            } else {
                dp[j] = Math.max(1, dp[j + 1] - dungeon[row - 1][j]);
            }
        }
        for (int i = row - 2; i >= 0; i--) {
            for (int j = column - 1; j >= 0; j--) {
                if (j == column - 1) {
                    dp[j] = Math.max(1, dp[j] - dungeon[i][j]);
                } else {
                    dp[j] = Math.max(1, Math.min(dp[j], dp[j + 1]) - dungeon[i][j]);
                }
            }
        }
        return dp[0];
    }


    public String largestNumber(int[] nums) {
        String[] values = new String[nums.length];
        for (int i = 0; i < nums.length; i++) {
            values[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(values, (o1, o2) -> {
            String value1 = o1 + o2;
            String value2 = o2 + o1;
            return value2.compareTo(value1);
        });
        if (values[0].equals("0")) {
            return "0";
        }
        StringBuilder builder = new StringBuilder();
        for (String value : values) {
            builder.append(value);
        }
        return builder.toString();
    }

    /**
     * @param str
     * @return
     */
    public char[] reverseWords(char[] str) {
        // write your code here


        int index = 0;

        while (index < str.length) {

            int current = index;

            while (current < str.length && str[current] != ' ') {
                current++;
            }
            swapWord(str, index, current - 1);

            index = current + 1;
        }

        swapWord(str, 0, str.length - 1);

        return str;
    }

    private void swapWord(char[] word, int start, int end) {
        for (int i = start; i <= (start + end) / 2; i++) {
            swap(word, i, (start + end) - i);
        }
    }

    private void swap(char[] str, int i, int j) {
        char word = str[i];
        str[i] = str[j];
        str[j] = word;
    }


    /**
     * 187. Repeated DNA Sequences
     *
     * @param s
     * @return
     */
    public List<String> findRepeatedDnaSequences(String s) {
        return null;
    }


    /**
     * 188. Best Time to Buy and Sell Stock IV
     *
     * @param k
     * @param prices
     * @return
     */
    public int maxProfit(int k, int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        if (k >= prices.length * 2) {
            return sellStock(prices);
        }
        int[][] dp = new int[k + 1][prices.length];
        for (int i = 1; i <= k; i++) {
            int tmp = -prices[0];
            for (int j = 1; j < prices.length; j++) {
                dp[i][j] = Math.max(dp[i][j - 1], prices[j] + tmp);
                tmp = Math.max(dp[i - 1][j - 1] - prices[j], tmp);
            }
        }
        return dp[k][prices.length - 1];
    }

    private int sellStock(int[] prices) {
        int result = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                result += prices[i] - prices[i - 1];
            }
        }
        return result;

    }


    public void rotate(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return;
        }
        k %= nums.length;
        k = nums.length - k;
        swapArray(nums, 0, k - 1);
        swapArray(nums, k, nums.length - 1);
        swapArray(nums, 0, nums.length - 1);
    }

    private void swapArray(int[] nums, int start, int end) {
        for (int i = start; i <= (start + end) / 2; i++) {
            swapValue(nums, i, start + end - i);
        }
    }

    private void swapValue(int[] nums, int i, int j) {
        int value = nums[i];
        nums[i] = nums[j];
        nums[j] = value;
    }


    /**
     * todo 190. Reverse Bits
     *
     * @param n
     * @return
     */
    public int reverseBits(int n) {
        return -1;
    }

    /**
     * 198. House Robber
     *
     * @param nums
     * @return
     */
    public int rob(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length + 1];

        dp[1] = nums[0];

        for (int i = 2; i < dp.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i - 1]);
        }
        return dp[nums.length];
    }


    /**
     * todo
     *
     * @param nums
     * @return
     */
    public int robV2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int robPrevious = 0;

        int robCurrent = 0;

        for (int i = 0; i < nums.length; i++) {

            int tmp = robPrevious;

            robPrevious = Math.max(robPrevious, robCurrent);

            robCurrent = tmp + nums[i];
        }
        return Math.max(robPrevious, robCurrent);
    }


    public List<Integer> rightSideView(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        LinkedList<TreeNode> linkedList = new LinkedList<>();

        linkedList.add(root);

        while (!linkedList.isEmpty()) {

            int size = linkedList.size();

            for (int i = 0; i < size; i++) {

                TreeNode node = linkedList.pollFirst();

                if (node.left != null) {
                    linkedList.add(node.left);
                }
                if (node.right != null) {
                    linkedList.add(node.right);
                }
                if (i == size - 1) {
                    result.add(node.val);
                }
            }
        }
        return result;

    }


    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (grid[i][j] == '1') {
                    intervalIslands(i, j, grid);
                    count++;
                }
            }
        }
        return count;

    }

    private void intervalIslands(int i, int j, char[][] grid) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[i].length) {
            return;
        }
        if (grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '0';
        intervalIslands(i - 1, j, grid);
        intervalIslands(i + 1, j, grid);
        intervalIslands(i, j - 1, grid);
        intervalIslands(i, j + 1, grid);
    }


}
