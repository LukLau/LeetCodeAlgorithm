package org.code.algorithm.leetcode.vip;

import org.code.algorithm.datastructe.TreeNode;

import java.util.*;

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

//        vip.upsideDownBinaryTreeV2(root);
//        int[] nums = new int[]{0, 1, 3, 50, 75};

        int[] nums = new int[]{2147483647};

        int lower = 0;

        int upper = Integer.MAX_VALUE;

        List<String> missingRanges = vip.findMissingRanges(nums, lower, upper);

//        System.out.println(missingRanges);


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
        HashMap<Character, Integer> map = new HashMap<>();

        int left = 0;

        int result = 0;

        int length = s.length();

        for (int i = 0; i < length; i++) {
            char word = s.charAt(i);

            Integer number = map.getOrDefault(word, 0);

            number++;

            map.put(word, number);

            while (map.size() > 2) {

                char leftWord = s.charAt(left);

                Integer leftWordNum = map.get(leftWord);

                leftWordNum--;

                map.put(leftWord, leftWordNum);

                if (leftWordNum == 0) {
                    map.remove(leftWord);
                }
                left++;
            }

            result = Math.max(result, i - left + 1);
        }
        return result;
    }


    public int lengthOfLongestSubstringTwoDistinctV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;

        Map<Character, Integer> map = new HashMap<>();

        int len = s.length();

        int left = 0;


        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);

            map.put(word, i);

            while (map.size() > 2) {
                Integer index = map.get(s.charAt(left));

                if (index == left) {
                    map.remove(s.charAt(left));
                }
                left++;
            }
            result = Math.max(result, i - left + 1);
        }
        return result;
    }

    public boolean isOneEditDistance(String s, String t) {
        // write your code here
        if (s == null || t == null) {
            return false;
        }
        if (s.equals(t)) {
            return false;
        }
        int m = s.length();
        int n = t.length();
        int diff = Math.abs(m - n);
        if (diff > 1) {
            return false;
        }
        if (m < n) {
            return isOneEditDistance(t, s);
        }
        if (diff == 1) {
            for (int i = 0; i < n; i++) {
                if (s.charAt(i) != t.charAt(i)) {
                    return s.substring(i + 1).equals(t.substring(i));
                }

            }
        }
        int diffWord = 0;

        for (int i = 0; i < n; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                diffWord++;
            }
            if (diffWord > 1) {
                return false;
            }
        }
        return true;
    }

    public boolean isOneEditDistanceV2(String s, String t) {
        if (s == null || t == null) {
            return false;
        }
        int m = s.length();

        int n = t.length();

        for (int i = 0; i < Math.min(m, n); i++) {
            if (s.charAt(i) != t.charAt(i)) {
                if (m == n) {
                    return s.substring(i + 1).equals(t.substring(i + 1));
                } else if (m < n) {
                    return s.substring(i).equals(t.substring(i + 1));
                } else {
                    return s.substring(i + 1).equals(t.substring(i));
                }
            }
        }
        return Math.abs(m - n) == 1;
    }


    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    /**
     * 163 丢失的范围 missing range
     *
     * @param nums
     * @param lower
     * @param upper
     * @return
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        if (nums == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        if (nums.length == 0) {
            result.add(findRange(lower, upper));
            return result;
        }
        if (nums[0] == lower + 1) {
            result.add(findRange(lower, nums[0] - 1));
        }
        for (int i = 0; i <= nums.length; i++) {

            long currentValue = i == nums.length ? (long) upper + 1 : nums[i];

            if (currentValue > lower + 1) {

                String range = findRange(lower + 1, currentValue - 1);

                result.add(range);
            }
            lower = (int) currentValue;
        }
        return result;
        // write your code here
        // write your code here
    }

    private String findRange(long lower, long upper) {
        return lower == upper ? upper + "" : lower + "->" + upper;
    }


}
