package org.code.algorithm.page;

import org.code.algorithm.datastructe.ListNode;

import java.util.HashMap;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/6/25
 */
public class FirstPage {


    private int beginPoint = 0;
    private int longestLen = Integer.MIN_VALUE;

    /**
     * 1. Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{};
        }
        HashMap<Integer, Integer> map = new HashMap<>();

        int[] result = new int[2];
        for (int i = 0; i < nums.length; i++) {
            int num = nums[i];
            if (map.containsKey(target - num)) {
                Integer index0 = map.get(target - num);
                result[0] = index0;
                result[1] = i;
                break;
            }
            map.put(num, i);
        }
        return result;
    }

    /**
     * 2. Add Two Numbers
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        int carry = 0;

        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (l1 != null || l2 != null || carry != 0) {
            int val = (l1 == null ? 0 : l1.val) + (l2 == null ? 0 : l2.val) + carry;

            ListNode node = new ListNode(val % 10);

            carry = val / 10;

            dummy.next = node;

            dummy = dummy.next;

            l1 = l1 == null ? null : l1.next;
            l2 = l2 == null ? null : l2.next;
        }
        return root.next;
    }

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        // todo
        return 0;
    }

    /**
     * 5. Longest Palindromic Substring
     *
     * @param s
     * @return
     */
    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int length = s.length();
        for (int i = 0; i < length; i++) {
            intervalPalindrome(s, i, i);
            intervalPalindrome(s, i, i + 1);
        }
        if (longestLen != Integer.MIN_VALUE) {
            return s.substring(beginPoint, beginPoint + longestLen);
        }
        return "";
    }


    private void intervalPalindrome(String s, int j, int k) {
        int length = s.length();
        while (j >= 0 && k < length && s.charAt(j) == s.charAt(k)) {
            if (s.charAt(j) == s.charAt(k) && k - j + 1 > longestLen) {
                longestLen = k - j + 1;
                beginPoint = j;
            }
            j--;
            k++;
        }
    }

    public String longestPalindromeV2(String s) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        int length = s.length();
        int maxLen = Integer.MIN_VALUE;
        int begin = 0;
        boolean[][] dp = new boolean[length][length];
        for (int i = 0; i < length; i++) {
            for (int j = 0; j <= i; j++) {
                if (s.charAt(j) == s.charAt(i)) {
                    if (i - j <= 2) {
                        dp[j][i] = true;
                    } else {
                        dp[j][i] = dp[j + 1][i - 1];
                    }
                }
                if (dp[j][i] && i - j + 1 > maxLen) {
                    begin = j;
                    maxLen = i - j + 1;
                }
            }
        }
        if (maxLen != Integer.MIN_VALUE) {
            return s.substring(begin, begin + maxLen);
        }
        return "";
    }


    /**
     * 10. Regular Expression Matching
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        if (p.isEmpty()) {
            return !s.isEmpty();
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;

        for (int i = 1; i <= n; i++) {
            dp[0][i] = p.charAt(i - 1) == '*' && dp[0][i - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                    }

                }
            }
        }
        return dp[m][n];
    }

    public boolean isMatchV2(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        boolean firstMatch = !s.isEmpty() && (s.charAt(0) == p.charAt(0) ||
                p.charAt(0) == '.');
        
        if (p.length() >= 2 && p.charAt(1) == '*') {
            return isMatchV2(s, p.substring(2)) || (firstMatch && isMatchV2(s.substring(1), p));
        }
        return firstMatch && isMatchV2(s.substring(1), p.substring(1));


    }


}
