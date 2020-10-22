package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;

/**
 * @author luk
 * @date 2020/10/21
 */
public class ThreePage {

    public static void main(String[] args) {
        ThreePage page = new ThreePage();

        String s = "egg";
        String t = "paa";

        page.isIsomorphic(s, t);
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


}
