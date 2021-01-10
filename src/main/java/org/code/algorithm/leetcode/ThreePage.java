package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author luk
 * @date 2020/10/21
 */
public class ThreePage {

    public static void main(String[] args) {
        ThreePage page = new ThreePage();

        int[] nums = new int[]{3, 1, 0, -2};

        int[] sort = new int[]{3, 5, 2, 1, 6, 4};

        int[][] board = new int[][]{{0, 1, 0}, {0, 0, 1}, {1, 1, 1}, {0, 0, 0}};

        int[] largest = new int[]{3, 2, 1, 5, 6, 4};
        int[] three = new int[]{-2, 0, -1, 3};
        page.threeSumSmaller(three, 2);
    }


    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        if (head.val == val) {
            return removeElements(head.next, val);
        }
        head.next = removeElements(head.next, val);
        return head;
    }


    /**
     * 204. Count Primes
     *
     * @param n
     * @return
     */
    public int countPrimes(int n) {
        return -1;
    }


    /**
     * 205. Isomorphic Strings
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isIsomorphic(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return true;
        }
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        int[] hash1 = new int[256];
        int[] hash2 = new int[256];
        for (int i = 0; i < m; i++) {
            if (hash1[s.charAt(i)] != hash2[t.charAt(i)]) {
                return false;
            }
            hash1[s.charAt(i)] = i + 1;

            hash2[t.charAt(i)] = i + 1;
        }
        return true;

    }

    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;

            prev = head;
            head = tmp;
        }
        return prev;
    }

    /**
     * 209. Minimum Size Subarray Sum
     */
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = Integer.MAX_VALUE;
        int end = 0;
        int value = 0;
        int begin = 0;
        while (end < nums.length) {
            value += nums[end];

            while (value >= s) {
                result = Math.min(result, end - begin + 1);
                value -= nums[begin++];
            }
            end++;
        }
        return result == Integer.MAX_VALUE ? 0 : result;
    }


    /**
     * 215. Kth Largest Element in an Array
     *
     * @param nums
     * @param k
     * @return
     */
    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }

    public int findKthLargestV2(int[] nums, int k) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>(k);
        for (int num : nums) {
            if (priorityQueue.size() == k && priorityQueue.peek() <= num) {
                priorityQueue.poll();
            }
            if (priorityQueue.size() != k) {
                priorityQueue.offer(num);
            }
        }
        ArrayList<Integer> integers = new ArrayList<>(priorityQueue);
        return integers.get(0);

    }


    /**
     * 217. Contains Duplicate
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (map.containsKey(num)) {
                return false;
            }
            map.put(num, 1);
        }
        return false;
    }


    /**
     * 218. The Skyline Problem
     * todo
     *
     * @param buildings
     * @return
     */
    public List<List<Integer>> getSkyline(int[][] buildings) {
        return null;
    }


    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {

            int val = nums[i];

            List<Integer> tmp = map.getOrDefault(val, new ArrayList<>());
            for (Integer j : tmp) {
                if (i - j <= k) {
                    return true;
                }
            }
            tmp.add(i);
            map.put(val, tmp);
        }
        return false;
    }


    /**
     * todo time exceed limit need bucket sort
     * 220. Contains Duplicate III
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (Math.abs(i - j) <= k) {
                    long diff = (long) nums[i] - (long) nums[j];
                    if (Math.abs(diff) <= t) {
                        return true;
                    }
                }
            }
        }
        return false;
    }


    /**
     * 221. Maximal Square
     *
     * @param matrix
     * @return
     */
    public int maximalSquare(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int[][] dp = new int[row][column];
        int result = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0) {
                        dp[0][j] = 1;


                    } else if (j == 0) {
                        dp[i][0] = 1;
                    } else {
                        dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                }
            }
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    result = Math.max(result, dp[i][j] * dp[i][j]);
                }
            }
        }
        return result;
    }

    /**
     * 222. Count Complete Tree Nodes
     *
     * @param root
     * @return
     */
    public int countNodes(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + countNodes(root.left) + countNodes(root.right);
    }

    /**
     * todo
     * 223. Rectangle Area
     *
     * @param A
     * @param B
     * @param C
     * @param D
     * @param E
     * @param F
     * @param G
     * @param H
     * @return
     */
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {

        int areaA = (C - A) * (D - B);
        return -1;
    }


    /**
     * 228. Summary Ranges
     *
     * @param nums
     * @return
     */
    public List<String> summaryRanges(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        if (nums.length == 1) {
            result.add(String.valueOf(nums[0]));
            return result;
        }
        int prev = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1] + 1) {
                String tmp = constructRange(nums, prev, i - 1);
                result.add(tmp);
                prev = i;
            }
        }
        result.add(constructRange(nums, prev, nums.length - 1));
        return result;
    }

    private String constructRange(int[] nums, int start, int end) {
        return start == end ? String.valueOf(nums[start]) : nums[start] + "->" + nums[end];
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode split = slow.next;
        slow.next = null;

        ListNode reverseNode = reverseNode(split);


        while (reverseNode != null) {
            if (reverseNode.val != head.val) {
                return false;
            }
            reverseNode = reverseNode.next;
            head = head.next;
        }
        return true;
    }

    private ListNode reverseNode(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode tmp = head.next;
            head.next = prev;
            prev = head;

            head = tmp;
        }
        return prev;
    }

    // ---公共祖先问题--- //

    /**
     * 235. Lowest Common Ancestor of a Binary Search Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q) {
            return root;
        }
        if (root.val > p.val && root.val > q.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (root.val < p.val && root.val < q.val) {
            return lowestCommonAncestor(root.right, p, q);
        } else {
            return root;
        }
    }

    /**
     * 236. Lowest Common Ancestor of a Binary Tree
     *
     * @param root
     * @param p
     * @param q
     * @return
     */
    public TreeNode lowestCommonAncestorII(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null) {
            return root;
        } else if (left == null) {
            return right;
        } else {
            return left;
        }
    }


    /**
     * 238. Product of Array Except Self
     *
     * @param nums
     * @return
     */
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        int len = nums.length;
        int[] products = new int[nums.length];
        int base = 1;
        for (int i = 0; i < products.length; i++) {
            products[i] = base;
            base *= nums[i];
        }
        base = 1;
        for (int i = products.length - 1; i >= 0; i--) {
            products[i] *= base;
            base *= nums[i];
        }
        return products;
    }


    /**
     * todo
     * 241. Different Ways to Add Parentheses
     *
     * @param input
     * @return
     */
    public List<Integer> diffWaysToCompute(String input) {
        if (input == null || input.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        char[] words = input.toCharArray();
        int end = 0;
        while (end < words.length) {
            if (Character.isDigit(words[end])) {
                int num = 0;
                while (end < words.length && Character.isDigit(words[end])) {
                    num = num * 10 + Character.getNumericValue(words[end]);
                    end++;
                }
                result.add(String.valueOf(num));
            } else {
                result.add(String.valueOf(words[end]));
                end++;
            }
        }
        return intervalDiffWays(result, 0, result.size() - 1);
    }

    private List<Integer> intervalDiffWays(List<String> params, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (start == end) {
            result.add(Integer.parseInt(params.get(start)));
            return result;
        }
        if (start > end) {
            return result;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            List<Integer> leftNums = intervalDiffWays(params, start, i - 1);
            List<Integer> rightNums = intervalDiffWays(params, i + 1, end);
            String sign = params.get(i);
            for (Integer leftNum : leftNums) {
                for (Integer rightNum : rightNums) {
                    if ("+".equals(sign)) {
                        result.add(leftNum + rightNum);
                    } else if ("-".equals(sign)) {
                        result.add(leftNum - rightNum);
                    } else if ("*".equals(sign)) {
                        result.add(leftNum * rightNum);
                    } else {
                        result.add(leftNum / rightNum);
                    }
                }
            }
        }
        return result;
    }


    /**
     * 242. Valid Anagram
     *
     * @param s
     * @param t
     * @return
     */
    public boolean isAnagram(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s.charAt(i) - 'a']++;
            hash[t.charAt(i) - 'a']--;
        }
        for (int count : hash) {
            if (count != 0) {
                return false;
            }
        }
        return true;
    }


    /**
     * 246 Strobogrammatic Number
     *
     * @param num: a string
     * @return: true if a number is strobogrammatic or false
     */
    public boolean isStrobogrammatic(String num) {
        // write your code here
        if (num == null || num.isEmpty()) {
            return false;
        }
        Map<Character, Character> map = new HashMap<>();
        map.put('0', '0');
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');


        char[] words = num.toCharArray();

        StringBuilder builder = new StringBuilder();
        for (char word : words) {
            if (!map.containsKey(word)) {
                return false;
            }
            builder.append(map.get(word));
        }
        String reverse = builder.reverse().toString();

        return num.equals(reverse);
    }


    /**
     * 247 Strobogrammatic Number II
     *
     * @param n: the length of strobogrammatic number
     * @return: All strobogrammatic numbers
     */
    public List<String> findStrobogrammatic(int n) {
        // write your code here
        if (n <= 0) {
            return Arrays.asList("");
        }
        char[] even = new char[]{'0', '1', '6', '8', '9'};
        char[] odd = new char[]{'0', '1', '8'};

        Map<Character, Character> map = getMap();
        List<String> tmp = new ArrayList<>();
        intervalFindStrobogrammatic(tmp, even, n / 2, "");

        List<String> result = new ArrayList<>();

        boolean isOdd = n % 2 == 1;

        for (String item : tmp) {
            char[] words = item.toCharArray();
            StringBuilder builder = new StringBuilder();
            for (char word : words) {
                builder.append(map.get(word));
            }
            String reverse = builder.reverse().toString();
            if (isOdd) {
                for (char oddChar : odd) {
                    result.add(item + oddChar + reverse);
                }
            } else {
                result.add(item + reverse);
            }
        }
        return result;
    }


    public List<String> findStrobogrammaticII(int n) {
        if (n == 0) {
            return Arrays.asList("");
        }
        if (n == 1) {
            return Arrays.asList("0", "1", "8");
        }
        return intervalFind(n, n);
    }

    private List<String> intervalFind(int m, int n) {
        if (m == 0) {
            return Collections.singletonList("");
        }
        if (m == 1) {
            return Arrays.asList("0", "1", "8");
        }
        List<String> result = new ArrayList<>();

        List<String> items = intervalFind(m - 2, n);

        for (String item : items) {
            if (m != n) {
                result.add("0" + item + "0");
            }
            result.add("1" + item + "1");
            result.add("6" + item + "9");
            result.add("8" + item + "8");
            result.add("9" + item + "6");
        }
        return result;
    }


    private void intervalFindStrobogrammatic(List<String> result, char[] items, int n, String s) {
        if (s.length() == n && !s.startsWith("0")) {
            result.add(s);
            return;
        }
        if (s.length() < n) {
            for (char item : items) {
                if (!s.startsWith("0")) {
                    intervalFindStrobogrammatic(result, items, n, s + String.valueOf(item));

                }
            }
        }
    }

    private Map<Character, Character> getMap() {
        HashMap<Character, Character> hashMap = new HashMap<>();
        hashMap.put('0', '0');
        hashMap.put('1', '1');
        hashMap.put('6', '9');
        hashMap.put('8', '8');
        hashMap.put('9', '6');
        return hashMap;
    }


    /**
     * https://www.cnblogs.com/grandyang/p/5203228.html
     * todo
     * 248 Strobogrammatic Number III
     *
     * @param low
     * @param high
     * @return
     */
    public int strobogrammaticInRange(String low, String high) {
        int count = 0;
        count += intervalStrobogrammaticInRange("", low, high);
        count += intervalStrobogrammaticInRange(0 + "" + 0, low, high);
        count += intervalStrobogrammaticInRange(1 + "" + 1, low, high);
        count += intervalStrobogrammaticInRange(8 + "" + 8, low, high);
        return count;
    }

    private int intervalStrobogrammaticInRange(String s, String low, String high) {
        int count = 0;

        int lowLen = low.length();
        int highLen = high.length();
        int len = s.length();
        if (len < lowLen || len > highLen) {
            return count;
        }
        if (len == highLen && s.compareTo(high) > 0) {
            return count;
        }
        if (len == 1 && s.startsWith("0") || len == lowLen && s.compareTo(low) < 0) {
            return count;
        }
        count++;
        if (len + 2 > highLen) {
            return count;
        }
        count += intervalStrobogrammaticInRange(0 + s + 0, low, high);
        count += intervalStrobogrammaticInRange(1 + s + 1, low, high);
        count += intervalStrobogrammaticInRange(6 + s + 9, low, high);
        count += intervalStrobogrammaticInRange(8 + s + 8, low, high);
        count += intervalStrobogrammaticInRange(9 + s + 6, low, high);
        return count;
    }


    /**
     * todo
     * 250 Count Univalue Subtrees
     *
     * @param root: the given tree
     * @return: the number of uni-value subtrees.
     */
    public int countUnivalSubtrees(TreeNode root) {
        // write your code here
        return -1;
    }


    /**
     * 259 3Sum Smaller
     * Medium
     *
     * @param nums:   an array of n integers
     * @param target: a target
     * @return: the number of index triplets satisfy the condition nums[i] + nums[j] + nums[k] < target
     */
    public int threeSumSmaller(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < nums.length - 2; i++) {
//            if (i > 0 && nums[i] == nums[i - 1]) {
//                continue;
//            }
            int left = i + 1;

            int right = nums.length - 1;

            while (left < right) {
                int val = nums[i] + nums[left] + nums[right];
                if (val < target) {
                    count += right - left;
                    left++;
                } else {
                    right--;
                }
            }
        }
        return count;
    }


    /**
     * todo
     * 269 Alien Dictionary
     * Hard
     *
     * @param words: a list of words
     * @return: a string which is correct order
     */
    public String alienOrder(String[] words) {
        // Write your code here
        return null;
    }


    /**
     * 272
     * Closest Binary Search Tree Value II
     *
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
        PriorityQueue<Integer> result = new PriorityQueue<>();
        TreeNode p = root;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            if (result.size() < k) {
                result.add(p.val);
            } else if ((Math.abs(p.val - target) < Math.abs(result.peek() - target))) {
                result.poll();
                result.offer(p.val);
            }
            p = p.right;
        }
        return new ArrayList<>(result);
    }


    //---H-index系列---//

    /**
     * 274. H-Index
     *
     * @param citations
     * @return
     */
    public int hIndex(int[] citations) {
        if (citations == null || citations.length == 0) {
            return 0;
        }
        int indexValue = 0;
        int indexCount = 1;
        for (int i = 0; i < citations.length; i++) {
            int current = citations[i];
            if (current >= indexValue) {
                indexCount++;
            }
        }
        return indexCount;
    }


    /**
     * todo 数学归纳法
     * 276 paint Fence
     *
     * @param n: non-negative integer, n posts
     * @param k: non-negative integer, k colors
     * @return: an integer, the total number of ways
     */
    public int numWays(int n, int k) {
        int[][] dp = new int[n + 1][k + 1];

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (i == 1) {
                    dp[i][j] = j;
                } else if (j == 1) {
                    dp[i][j] = i;
                }

            }
        }
        return -1;
    }


    /**
     * 280 Wiggle Sort
     *
     * @param nums
     */
    public void wiggleSort(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        Arrays.sort(nums);
        for (int i = 1; i < nums.length - 1; i = i + 2) {
            swapValue(nums, i, i + 1);
        }
    }

    private void swapValue(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    /**
     * 286 Walls and Gates
     *
     * @param rooms: m x n 2D grid
     * @return: nothing
     */
    public void wallsAndGates(int[][] rooms) {
        if (rooms == null || rooms.length == 0) {
            return;
        }
        int row = rooms.length;
        int column = rooms[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (rooms[i][j] == 0) {
                    intervalWallsAndGates(rooms, i, j, 0);
                }
            }
        }

    }

    private void intervalWallsAndGates(int[][] rooms, int i, int j, int distance) {
        if (i < 0 || i >= rooms.length || j < 0 || j >= rooms[i].length) {
            return;
        }
        if (rooms[i][j] == -1) {
            return;
        }
        if (rooms[i][j] > distance || rooms[i][j] == distance) {
            rooms[i][j] = distance;
            intervalWallsAndGates(rooms, i - 1, j, distance + 1);
            intervalWallsAndGates(rooms, i + 1, j, distance + 1);
            intervalWallsAndGates(rooms, i, j - 1, distance + 1);
            intervalWallsAndGates(rooms, i, j + 1, distance + 1);
        }
    }


    /**
     * todo
     * 287. Find the Duplicate Number
     *
     * @param nums
     * @return
     */
    public int findDuplicate(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        return -1;
    }


    /**
     * 289. Game of Life
     *
     * @param board: the given board
     * @return: nothing
     */
    public void gameOfLife(int[][] board) {
        // Write your code here
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                intervalGameOfLife(i, j, board);
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] > 0) {
                    board[i][j] = 1;
                } else {
                    board[i][j] = 0;
                }
            }
        }
    }

    private void intervalGameOfLife(int currentRow, int currentColumn, int[][] board) {
        int liveCount = 0;
        int[][] matrix = new int[][]{{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
        for (int i = 0; i < matrix.length; i++) {
            int row = currentRow + matrix[i][0];
            int column = currentColumn + matrix[i][1];
            if (row < 0 || row >= board.length || column < 0 || column >= board[currentRow].length) {
                continue;
            }
            if (Math.abs(board[row][column]) == 1) {
                liveCount++;
            }
        }
//        int[][] line = new int[][]{z};
//        for (int i = 0; i < line.length; i++) {
//            int row = currentRow + line[i][0];
//            int column = currentColumn + line[i][1];
//            if (row < 0 || row >= board.length || column < 0 || column >= board[currentRow].length) {
//                continue;
//            }
//            if (board[row][column] == 1) {
//                liveCount++;
//            }
//        }
        boolean currentLive = board[currentRow][currentColumn] == 1;

        if (currentLive && (liveCount < 2 || liveCount > 3)) {
            board[currentRow][currentColumn] = -1;
        }
        if (board[currentRow][currentColumn] == 0 && liveCount == 3) {
            // 2 signifies the cell is now live but was originally dead.
            board[currentRow][currentColumn] = 2;
        }
    }


    /**
     * 290. Word Pattern
     *
     * @param pattern
     * @param str
     * @return
     */
    public boolean wordPattern(String pattern, String str) {
        String[] words = str.split(" ");

        int length = pattern.length();
        if (length != words.length) {
            return false;
        }
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < words.length; i++) {
            if (!Objects.equals(map.put(String.valueOf(pattern.charAt(i)), i), map.put(words[i], i))) {
                return false;
            }
        }
        return true;
    }


    /**
     * 290.
     * todo
     *
     * @param pattern: a string,denote pattern string
     * @param str:     a string, denote matching string
     * @return: a boolean
     */
    public boolean wordPatternMatch(String pattern, String str) {
        // write your code here
        Map<Character, String> map = new HashMap<>();
        return intervalWordPatternMatch(pattern, str, map);
    }

    private boolean intervalWordPatternMatch(String pattern, String str, Map<Character, String> map) {
        if (pattern.length() == 0) {
            return str.length() == 0;
        }
        char ch = pattern.charAt(0);
        if (map.containsKey(ch)) {
            String word = map.get(ch);
            if (!str.startsWith(word)) {
                return false;
            }
            return intervalWordPatternMatch(pattern.substring(1), str.substring(word.length()), map);
        }
        for (int i = 0; i < str.length(); i++) {
            String word = str.substring(0, i + 1);

            map.put(ch, word);

            if (intervalWordPatternMatch(pattern.substring(1), str.substring(i + 1), map)) {
                return true;
            }
            map.remove(ch);
        }
        return false;
    }


    /**
     * 293 Flip Game
     *
     * @param s: the given string
     * @return: all the possible states of the string after one valid move
     */
    public List<String> generatePossibleNextMoves(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        for (int i = 0; i < s.length() - 1; i++) {
            int index = s.indexOf("++", i);
            if (index == -1) {
                return result;
            }
            StringBuilder builder = new StringBuilder();
            String word = s.substring(0, index);
            builder.append(word);
            builder.append("--");
            builder.append(s.substring(index + 2));
            result.add(builder.toString());
            i = index;
        }
        return result;
    }


    /**
     * todo
     * 294 Flip Game II
     *
     * @param s
     * @return
     */
    public boolean canWin(String s) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        List<String> result = generatePossibleNextMoves(s);
        return result.size() % 2 != 0;
    }


    public String getHint(String secret, String guess) {
        int bulls = 0;
        int crows = 0;
        int len = secret.length();
        int[] nums = new int[10];
        for (int i = 0; i < len; i++) {
            char s = secret.charAt(i);
            char g = guess.charAt(i);
            if (s == g) {
                bulls++;
            } else {
                if (nums[Character.getNumericValue(s)]-- > 0) {
                    crows++;
                }
                if (nums[Character.getNumericValue(g)]++ < 0) {
                    crows++;
                }
            }
        }
        return bulls + "A" + crows + "B";
    }


}
