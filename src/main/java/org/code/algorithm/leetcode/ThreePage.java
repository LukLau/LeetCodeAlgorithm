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

        page.maximalSquare(matrix);
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


}
