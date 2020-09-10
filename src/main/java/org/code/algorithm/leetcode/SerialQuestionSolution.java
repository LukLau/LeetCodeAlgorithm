package org.code.algorithm.leetcode;

import com.sun.jmx.remote.internal.ArrayQueue;
import org.code.algorithm.datastructe.ListNode;

import java.lang.reflect.Array;
import java.util.*;

/**
 * @author luk
 * @date 2020/9/8
 */
public class SerialQuestionSolution {


    // ---最长无重复子串问题--- //

    public static void main(String[] args) {
        SerialQuestionSolution solution = new SerialQuestionSolution();
        ListNode head = new ListNode(1);
        ListNode two = new ListNode(2);

        head.next = two;

        ListNode three = new ListNode(3);
        two.next = three;

        ListNode four = new ListNode(4);

        three.next = four;

        ListNode five = new ListNode(5);

        four.next = five;

        solution.reverseKGroupV2(head, 2);

    }

    // ---O log(N)算法---- //

    /**
     * 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int left = 0;
        int result = 0;
        int length = s.length();
        HashMap<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < length; i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            result = Math.max(result, i - left + 1);
        }
        return result;

    }


    /**
     * 34. Find First and Last Position of Element in Sorted Array
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int[] result = new int[2];
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (nums[left] != target) {
            return new int[]{-1, -1};
        }
        result[0] = left;
        right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2 + 1;
            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid;
            }
        }
        result[1] = left;

        return result;
    }


    public int[] searchRangeV2(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int firstIndex = searchLeftIndex(nums, target, 0, nums.length - 1);
        return nums;
    }


    public int searchInsert(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] > target) {
                left = mid + 1;
            } else {
                right = mid + 1;
            }
        }
        return left;

    }


    private int searchLeftIndex(int[] nums, int target, int left, int right) {
        if (left > right) {
            return -1;
        }
        int mid = left + (right - left) / 2;
        if (nums[mid] < target) {
            return searchLeftIndex(nums, target, mid + 1, right);
        } else if (nums[mid] > target) {
            return searchLeftIndex(nums, target, left, mid - 1);
        } else {
            if (mid - 1 > 0 && nums[mid] == nums[mid - 1]) {
                return searchLeftIndex(nums, target, left, mid - 1);
            }
            if (nums[mid] == target) {
                return mid;
            }
        }
        return -1;
    }

    // --正则表达式匹配问题 //

    /**
     * 4. Median of Two Sorted Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        return -1;
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
            return s.isEmpty();
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];

        dp[0][0] = true;

        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                    }
                }
            }
        }
        return dp[m][n];
    }

    // --旋转数组系列问题-- //

    /**
     * 30. Substring with Concatenation of All Words
     * todo
     * answer https://github.com/grandyang/leetcode/issues/30
     */
    public List<Integer> findSubstring(String s, String[] words) {
        return null;
    }

    /**
     * 33. Search in Rotated Sorted Array
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] < nums[mid]) {
                if (target < nums[mid] && nums[left] <= target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return nums[left] == target ? left : -1;
    }


    /**
     * 23. Merge k Sorted Lists
     *
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> priorityQueue = new PriorityQueue<>(lists.length, Comparator.comparingInt(o -> o.val));
        for (ListNode list : lists) {
            if (list != null) {
                priorityQueue.offer(list);
            }
        }
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        while (!priorityQueue.isEmpty()) {
            ListNode poll = priorityQueue.poll();

            dummy.next = poll;

            dummy = dummy.next;

            if (poll.next != null) {
                priorityQueue.offer(poll.next);
            }
        }
        return root.next;
    }


    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        int count = 0;
        ListNode currentNode = head;
        while (currentNode != null && count != k) {
            currentNode = currentNode.next;
            count++;
        }
        if (count == k) {
            ListNode reverseKGroup = reverseKGroup(currentNode, k);
            while (k-- > 0) {
                ListNode tmp = head.next;
                head.next = reverseKGroup;
                reverseKGroup = head;
                head = tmp;
            }
            head = reverseKGroup;
        }
        return head;
    }


    /**
     * todo
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroupV2(ListNode head, int k) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode currentNode = head;
        for (int i = 0; i < k; i++) {
            if (currentNode == null) {
                return head;
            }
            currentNode = currentNode.next;
        }

        while (head != currentNode) {
            ListNode tmp = head.next;
            head.next = currentNode;
            currentNode = head;
            head = tmp;
        }

        return currentNode;
    }

    // ---排列组合问题---- //

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        intervalCombinationSum(result, new ArrayList<Integer>(), 0, candidates, target);
        return result;


    }

    private void intervalCombinationSum(List<List<Integer>> result, ArrayList<Integer> integers, int start, int[] candidates, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            integers.add(candidates[i]);
            intervalCombinationSum(result, integers, i, candidates, target - candidates[i]);
            integers.remove(integers.size() - 1);
        }
    }


    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<>();
        intervalCombinationSum2(result, new ArrayList<>(), 0,candidates,target);
        return result;


    }

    private void intervalCombinationSum2(List<List<Integer>> result, ArrayList<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
//            result.add(new ArrayQueue<>(tmp));
            return;
        }
//        for (int i = start; i < )
    }
}
