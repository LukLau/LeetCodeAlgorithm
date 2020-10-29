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

        int[] nums = new int[]{1, 2, 3, 1};

        char[][] matrix = new char[][]{{'0', '1'}};

        System.out.println(page.findStrobogrammatic(3));
    }


    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        if (head.val == val) {
            while (head != null && head.val == val) {
                head = head.next;
            }
            return removeElements(head, val);
        }
        head.next = removeElements(head.next, val);

        return head;
    }


    public int countPrimes(int n) {

        if (n <= 1) {
            return 0;
        }
        for (int i = 2; i <= n; i++) {
            for (int j = 2; j < i; j++) {

            }
        }
        return -1;
    }


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
            int index1 = s.charAt(i);

            int index2 = t.charAt(i);

            if (hash1[index1] != hash2[index2]) {
                return false;
            }
            hash1[index1] = i + 1;

            hash2[index2] = i + 1;
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


    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        int kthLargestIndex = nums.length - k;
        return nums[kthLargestIndex];
    }


    /**
     * 217. Contains Duplicate
     *
     * @param nums
     * @return
     */
    public boolean containsDuplicate(int[] nums) {
        int[] distinctNums = Arrays.stream(nums).distinct().toArray();
        return nums.length != distinctNums.length;
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
            List<Integer> indexList = map.getOrDefault(val, new ArrayList<>());
            for (int j = indexList.size() - 1; j >= 0; j--) {
                int diff = i - indexList.get(j);

                if (diff <= k) {
                    return true;
                }
            }
            indexList.add(i);

            map.put(val, indexList);
        }
        return false;
    }


    /**
     * todo time exceed limit
     * 220. Contains Duplicate III
     *
     * @param nums
     * @param k
     * @param t
     * @return
     */
    public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = i - 1; j >= 0 && j >= i - k; j--) {
                long diff = Math.abs((long) nums[i] - (long) nums[j]);
                if (diff <= t) {
                    return true;
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
                char word = matrix[i][j];
                if (word == '0') {
                    continue;
                }
                if (i == 0) {
                    dp[i][j] = 1;
                } else if (j == 0) {
                    dp[i][j] = 1;
                } else {
                    int width = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;

                    dp[i][j] = width;
                }
                result = Math.max(result, dp[i][j] * dp[i][j]);
            }
        }
        return result;

    }

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
        int previousIndex = Integer.MAX_VALUE;

        for (int i = 0; i < nums.length; i++) {
            if (i == 0) {
                previousIndex = i;
            } else if (nums[i] != nums[i - 1] + 1) {
                String range = nums[i - 1] == nums[previousIndex] ? String.valueOf(nums[previousIndex]) : nums[previousIndex] + "->" + nums[i - 1];

                result.add(range);

                previousIndex = i;
            }
        }
        result.add(constructRange(nums, previousIndex, nums.length - 1));

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
        ListNode prev = slow;
        while (fast != null && fast.next != null) {
            prev = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        prev.next = null;

        ListNode reverseNode = reverseNode(slow);

        while (head != null) {
            if (head.val != reverseNode.val) {
                return false;
            }
            head = head.next;
            reverseNode = reverseNode.next;
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

        TreeNode leftNode = lowestCommonAncestorII(root.left, p, q);

        TreeNode rightNode = lowestCommonAncestorII(root.right, p, q);
        if (leftNode != null && rightNode != null) {
            return root;
        } else if (leftNode == null) {
            return rightNode;
        } else {
            return leftNode;
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
        int baseNum = 1;
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            result[i] = baseNum;
            baseNum *= nums[i];
        }
        baseNum = 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            result[i] *= baseNum;
            baseNum *= nums[i];
        }
        return result;
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
        List<String> params = new ArrayList<>();
        int len = input.length();
        int endIndex = 0;
        while (endIndex < len) {
            char tmp = input.charAt(endIndex);
            if (Character.isDigit(tmp)) {
                int value = 0;
                while (endIndex < len && Character.isDigit(input.charAt(endIndex))) {
                    value = value * 10 + Character.getNumericValue(input.charAt(endIndex++));
                }
                params.add(String.valueOf(value));
            } else {
                params.add(String.valueOf(tmp));
                endIndex++;
            }
        }
        return intervalDiffWays(params, 0, params.size() - 1);
    }

    private List<Integer> intervalDiffWays(List<String> params, int start, int end) {
        List<Integer> result = new ArrayList<>();
        if (start == end) {
            result.add(Integer.parseInt(params.get(start)));
            return result;
        }
        for (int i = start + 1; i <= end - 1; i = i + 2) {
            String sign = params.get(i);
            List<Integer> leftNums = intervalDiffWays(params, start, i - 1);
            List<Integer> rightNums = intervalDiffWays(params, i + 1, end);
            for (Integer leftNum : leftNums) {
                for (Integer rightNum : rightNums) {

                    if ("+".equals(sign)) {
                        result.add(leftNum + rightNum);
                    } else if ("-".equals(sign)) {
                        result.add(leftNum - rightNum);
                    } else if ("*".equals(sign)) {
                        result.add(leftNum * rightNum);
                    } else if ("/".equals(sign)) {
                        result.add(leftNum / rightNum);
                    }
                }
            }
        }
        return result;
    }


    public boolean isAnagram(String s, String t) {
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
        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s.charAt(i) - 'a']--;
            hash[t.charAt(i) - 'a']++;
        }
        for (int num : hash) {
            if (num != 0) {
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
        if (num.length() == 1) {
            return "0".equals(num) || "8".equals(num) || "1".equals(num);
        }
        Map<Character, Character> map = getMap();
        int len = num.length();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < len; i++) {
            char word = num.charAt(i);
            if (!map.containsKey(word)) {
                return false;
            }
            builder.append(map.get(word));
        }
        return builder.reverse().toString().equals(num);
    }


    /**
     * 247 Strobogrammatic Number II
     *
     * @param n: the length of strobogrammatic number
     * @return: All strobogrammatic numbers
     */
    public List<String> findStrobogrammatic(int n) {
        // write your code here
        if (n == 0) {
            return Arrays.asList("");
        }
        if (n == 1) {
            return Arrays.asList("0", "1", "8");
        }
        Map<Character, Character> map = getMap();
        char[] nums = new char[]{'0', '1', '6', '8', '9'};
        char[] odds = new char[]{'0', '1', '8'};
        List<String> result = new ArrayList<>();
        intervalFindStrobogrammatic(result, nums, n / 2, "");
        List<String> ans = new ArrayList<>();
        boolean isOdd = n % 2 != 0;
        for (String s : result) {
            char[] chars = s.toCharArray();
            String reverse = "";
            for (char word : chars) {
                Character character = map.get(word);
                reverse += character;
            }
            reverse = new StringBuilder(reverse).reverse().toString();
            if (isOdd) {
                for (char odd : odds) {
                    String tmp = s + odd + reverse;
                    ans.add(tmp);
                }
            } else {
                s += reverse;
                ans.add(s);
            }
        }
        return ans;
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
        if (n == 0) {
            return Collections.singletonList("");
        }
        if (n == 1) {
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
                    intervalFindStrobogrammatic(result, items, n, s + item);
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
        return -1;
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


}
