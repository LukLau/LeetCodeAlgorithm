package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;

import java.util.*;

/**
 * @author luk
 * @date 2020/9/8
 */
public class SerialQuestionSolution {


    // ---最长无重复子串问题--- //

    public static void main(String[] args) {
        SerialQuestionSolution solution = new SerialQuestionSolution();
//        ListNode head = new ListNode(1);
//        ListNode two = new ListNode(2);
//
//        head.next = two;
//
//        ListNode three = new ListNode(3);
//        two.next = three;
//
//        ListNode four = new ListNode(4);
//
//        three.next = four;
//
//        ListNode five = new ListNode(5);
//
//        four.next = five;
//
//        solution.reverseKGroupV2(head, 2);

        int[][] matrix = new int[][]{{2, 3}, {2, 2}, {3, 3}, {1, 3}, {5, 7}, {2, 2}, {4, 6}};

        solution.searchV2(new int[]{1, 1}, 0);

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


    // --正则表达式匹配问题 //

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


    /**
     * 45. Jump Game II
     * todo
     *
     * @param s
     * @param p
     * @return
     */
    public boolean isMatchV2(String s, String p) {
        if (p.isEmpty()) {
            return s.isEmpty();
        }
        int m = s.length();
        int n = p.length();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int j = 1; j <= n; j++) {
            dp[0][j] = p.charAt(j - 1) == '*' && dp[0][j - 1];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int reach = 0;
        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {
            reach = Math.max(i + nums[i], reach);
        }
        return reach >= nums.length - 1;
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
     * 81. Search in Rotated Sorted Array II
     *
     * @param nums
     * @param target
     * @return
     */
    public boolean searchV2(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[left] == nums[right]) {
                left++;
            } else if (nums[left] <= nums[mid]) {
                if (target < nums[mid] && target >= nums[left]) {
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
        return nums[left] == target;
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
        intervalCombinationSum2(result, new ArrayList<>(), 0, candidates, target);
        return result;
    }

    private void intervalCombinationSum2(List<List<Integer>> result, ArrayList<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            intervalCombinationSum2(result, tmp, i + 1, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }


    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        intervalPermute(result, new ArrayList<Integer>(), used, nums);
        return result;

    }

    private void intervalPermute(List<List<Integer>> result, ArrayList<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            integers.add(nums[i]);
            used[i] = true;
            intervalPermute(result, integers, used, nums);
            used[i] = false;
            integers.remove(integers.size() - 1);
        }

    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        intervalPermuteUnique(result, new ArrayList<Integer>(), used, nums);
        return result;
    }

    private void intervalPermuteUnique(List<List<Integer>> result, ArrayList<Integer> tmp, boolean[] used, int[] nums) {
        if (tmp.size() == nums.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            if (used[i]) {
                continue;
            }
            tmp.add(nums[i]);
            used[i] = true;
            intervalPermuteUnique(result, tmp, used, nums);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }

    }


    /**
     * todo 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        List<Integer> numbers = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            numbers.add(i);
        }
        k--;
        int[] position = new int[k + 1];
        position[0] = 1;
        position[1] = 1;

        return "";


    }

    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> result = new ArrayList<>();
        intervalCombine(result, new ArrayList<Integer>(), 1, n, k);
        return result;
    }

    private void intervalCombine(List<List<Integer>> result, ArrayList<Integer> integers, int start, int n, int k) {
        if (integers.size() == k) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i <= n; i++) {
            integers.add(i);
            intervalCombine(result, integers, i + 1, n, k);
            integers.remove(integers.size() - 1);
        }
    }

    public List<List<Integer>> subsets(int[] nums) {

        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalSubsets(result, new ArrayList<Integer>(), 0, nums);
        return result;

    }

    private void intervalSubsets(List<List<Integer>> result, ArrayList<Integer> integers, int start, int[] nums) {
        result.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            integers.add(nums[i]);
            intervalSubsets(result, integers, i + 1, nums);
            integers.remove(integers.size() - 1);
        }
    }


    // ---- //


    /**
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i--) {
            while (i >= 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return nums.length + 1;
    }

    private void swap(int[] nums, int i, int j) {
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }


    public int trap(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int leftEdge = 0;
        int rightEdge = 0;
        int result = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                if (height[left] >= leftEdge) {
                    leftEdge = height[left];
                } else {
                    result += leftEdge - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightEdge) {
                    rightEdge = height[right];
                } else {
                    result += rightEdge - height[right];
                }
                right--;
            }
        }
        return result;
    }

    public int trapV2(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int left = 0;
        int right = height.length - 1;
        int result = 0;
        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;
            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int tmp = Math.min(height[left], height[right]);

            for (int i = left; i <= right; i++) {
                if (height[i] >= tmp) {
                    height[i] -= tmp;
                } else {
                    result += tmp - height[i];
                    height[i] = 0;
                }
            }
        }
        return result;
    }


    public String multiply(String num1, String num2) {
        if (num1 == null && num2 == null) {
            return "0";
        }
        int m = num1.length();
        int n = num2.length();
        int[] nums = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int tmp = Character.getNumericValue(num1.charAt(i)) * Character.getNumericValue(num2.charAt(j)) + nums[i + j + 1];

                nums[i + j + 1] = tmp % 10;

                nums[i + j] += tmp / 10;
            }
        }
        StringBuilder builder = new StringBuilder();
        for (int num : nums) {
            if (!(builder.length() == 0 && num == 0)) {
                builder.append(num);
            }
        }
        return builder.length() == 0 ? "0" : builder.toString();
    }


    // ---跳跃格子游戏系列--- //


    public int jump(int[] nums) {
        int step = 0;
        int currentIndex = 0;
        int furthest = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            currentIndex = Math.max(nums[i] + i, currentIndex);
            if (i == furthest) {
                step++;
                furthest = currentIndex;
            }
        }
        return step;
    }


    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        int row = matrix.length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < i; j++) {
                swapMatrix(matrix, i, j);
            }
        }
        for (int[] ints : matrix) {
            for (int j = 0; j <= (ints.length - 1) / 2; j++) {
                swap(ints, j, (ints.length - 1) - j);
            }
        }
    }

    private void swapMatrix(int[][] matrix, int i, int j) {
        int val = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = val;
    }

    // --数学原理--- //

    public double myPow(double x, int n) {
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            n = -n;
            x = 1 / x;
        }
        if (x > Integer.MAX_VALUE || x < Integer.MIN_VALUE) {
            return 0;
        }
        return n % 2 != 0 ? x * myPow(x * x, n / 2) : myPow(x * x, n / 2);
    }


    /**
     * 牛顿二分法
     *
     * @param x
     * @return
     */
    public int mySqrt(int x) {
        double precision = 0.00001;
        double result = x;
        while (result * result - x > precision) {
            result = (result + x / result) / 2;
        }
        return (int) result;

    }


    /**
     * todo
     * 65. Valid Number
     *
     * @param s
     * @return
     */
    public boolean isNumber(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        return false;

    }


    // ---- 八皇后问题----//


    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<String>> result = new ArrayList<>();

        char[][] words = new char[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                words[i][j] = '.';
            }
        }
        intervalSolveNQueens(result, words, 0, n);
        return result;
    }

    private void intervalSolveNQueens(List<List<String>> result, char[][] words, int row, int n) {
        if (row == n) {
            List<String> tmp = new ArrayList<>();
            for (char[] word : words) {
                tmp.add(String.valueOf(word));
            }
            result.add(tmp);
            return;
        }
        for (int j = 0; j < n; j++) {
            if (validNQueens(words, row, j)) {
                words[row][j] = 'Q';
                intervalSolveNQueens(result, words, row + 1, n);
                words[row][j] = '.';
            }
        }

    }

    private boolean validNQueens(char[][] words, int row, int column) {
        for (int i = row - 1; i >= 0; i--) {
            if (words[i][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (words[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < words.length; i--, j++) {
            if (words[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }


    public int totalNQueens(int n) {
        if (n <= 0) {
            return 0;
        }
        int[] dp = new int[n];
        return intervalTotalNQueens(dp, 0, n);

    }

    private int intervalTotalNQueens(int[] dp, int row, int n) {
        if (row == n) {
            return 1;
        }
        int count = 0;
        for (int j = 0; j < n; j++) {
            if (validTotalNQueens(dp, row, j)) {
                dp[row] = j;
                count += intervalTotalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean validTotalNQueens(int[] dp, int row, int col) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == col || Math.abs(dp[i] - col) == Math.abs(i - row)) {
                return false;
            }
        }
        return true;
    }

    // --合并区间问题- //


    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        LinkedList<int[]> result = new LinkedList<>();
        for (int[] interval : intervals) {
            if (!result.isEmpty() && result.getLast()[1] >= interval[1]) {
                result.getLast()[1] = Math.max(result.getLast()[1], interval[1]);
            } else {
                result.offer(interval);
            }
        }
        return result.toArray(new int[][]{});
    }

    /**
     * 57. Insert Interval
     *
     * @param intervals
     * @param newInterval
     * @return
     */
    public int[][] insert(int[][] intervals, int[] newInterval) {
        if (intervals == null || intervals.length == 0 | newInterval == null || newInterval.length == 0) {
            return new int[][]{};
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        LinkedList<int[]> ans = new LinkedList<>();
        int index = 0;
        while (index < intervals.length && intervals[index][1] < newInterval[0]) {
            ans.offer(intervals[index++]);
        }
        while (index < intervals.length && intervals[index][0] <= newInterval[1]) {
            newInterval[0] = Math.min(intervals[index][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[index][1], newInterval[1]);

            index++;
        }
        ans.offer(newInterval);

        while (index < intervals.length) {
            ans.offer(intervals[index++]);
        }
        return ans.toArray(new int[][]{});
    }

    // --最小路径-- //

    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int[][] dp = new int[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (i == 0 && j == 0) {
                    dp[i][j] = grid[i][j];
                } else if (i == 0) {
                    dp[i][j] = dp[i][j - 1] + grid[i][j];
                } else if (j == 0) {
                    dp[i][j] = dp[i - 1][j] + grid[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                }
            }
        }
        return dp[row - 1][column - 1];
    }


    // --链表相关接口-- //

    /**
     * 61. Rotate List
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode current = head;
        int count = 1;
        while (current.next != null) {
            count++;
            current = current.next;
        }
        current.next = head;
        k %= count;
        if (k != 0) {
            for (int i = 0; i < count - k; i++) {
                current = current.next;
                head = head.next;
            }
        }
        current.next = null;
        return head;
    }


    /**
     * 68. Text Justification
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustify(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        int startIndex = 0;

        while (startIndex < words.length) {
            int endIndex = startIndex;
            int line = 0;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex].length() + 1;
                endIndex++;
            }
            boolean lastRow = endIndex == words.length;
            int countOfWord = endIndex - startIndex;
            int blankRow = maxWidth - line + 1;
            StringBuilder builder = new StringBuilder();
            if (countOfWord == 1) {
                builder.append(words[startIndex]);
            } else {
                int blankNum = lastRow ? 1 : 1 + blankRow / (countOfWord - 1);
                int extraBlankNum = lastRow ? 0 : blankRow % (countOfWord - 1);
                builder.append(constructRow(words, startIndex, endIndex, blankNum, extraBlankNum));
            }
            startIndex = endIndex;

            result.add(trimRow(builder.toString(), maxWidth));
        }
        return result;
    }

    private String trimRow(String word, int maxWidth) {
        while (word.length() < maxWidth) {
            word += " ";
        }
        while (word.length() > maxWidth) {
            word = word.substring(0, word.length() - 1);
        }
        return word;
    }

    private String constructRow(String[] words, int start, int end, int blankNum, int extraBlankNum) {
        StringBuilder result = new StringBuilder();
        for (int i = start; i < end; i++) {
            result.append(words[i]);
            int tmp = blankNum;
            while (tmp-- > 0) {
                result.append(" ");
            }
            if (extraBlankNum-- > 0) {
                result.append(" ");
            }
        }
        return result.toString();
    }


    public List<String> fullJustifyV2(String[] words, int maxWidth) {
        if (words == null || words.length == 0 || maxWidth <= 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        for (int i = 0, k; i < words.length; i += k) {
            k = 0;
            int line = 0;
            while (i + k < words.length && line + words[i + k].length() <= maxWidth - k) {
                line += words[i + k].length();
                k++;
            }
            boolean lastRow = i + k == words.length;

            int blankLine = maxWidth - line;

            int intervalCount = k - 1;

            StringBuilder builder = new StringBuilder();

            for (int j = 0; j < k; j++) {

                builder.append(words[i + j]);
                if (lastRow) {
                    builder.append(" ");
                } else {
                    int blankWord = blankLine / intervalCount + (j < blankLine % intervalCount ? 1 : 0);

                    while (blankWord-- > 0) {
                        builder.append(" ");
                    }
                }
            }
            result.add(trimRow(builder.toString(), maxWidth));
        }
        return result;
    }


    public String simplifyPath(String path) {
        if (path == null || path.isEmpty()) {
            return "/";
        }
        List<String> skip = Arrays.asList(".", "", "/");
        String[] words = path.split("/");
        LinkedList<String> ans = new LinkedList<>();
        for (String word : words) {
            if ("..".equals(word)) {
                ans.pollFirst();
            } else if (!skip.contains(word)) {
                ans.add(word);
            }
        }
        if (ans.isEmpty()) {
            return "/";
        }
        StringBuilder result = new StringBuilder();
        for (String word : ans) {
            result.append("/").append(word);
        }
        return result.toString();
    }

    // --编辑距离问题- //

    public int minDistance(String word1, String word2) {
        if (word1 == null || word2 == null) {
            return 0;
        }
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            dp[i][0] = i;
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[m][n];
    }


    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int i = 0;
        int j = column - 1;
        while (i < row && j >= 0) {
            int value = matrix[i][j];
            if (value == target) {
                return true;
            } else if (value < target) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }


    // ---双指针问题系列-- //


    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int redNum = 0;
        int blueNum = nums.length - 1;
        for (int i = 0; i < nums.length; i++) {
            while (i < blueNum && nums[i] == 2) {
                swap(nums, i, blueNum--);
            }
            while (i > redNum && nums[i] == 0) {
                swap(nums, i, redNum++);
            }
        }
    }


    // --滑动窗口问题- //

    public String minWindow(String s, String t) {
        if (s == null || t == null) {
            return "";
        }
        int result = Integer.MAX_VALUE;
        int begin = 0;
        int end = 0;
        int head = 0;
        int n = t.length();
        int m = s.length();
        int[] hash = new int[256];
        for (int i = 0; i < n; i++) {
            hash[t.charAt(i) - '0']++;
        }
        while (end < m) {
            if (hash[s.charAt(end++) - '0']-- > 0) {
                n--;
            }
            while (n == 0) {
                if (end - begin < result) {
                    result = end - begin;
                    head = begin;
                }
                if (hash[s.charAt(begin++) - '0']++ == 0) {
                    n++;
                }
            }
        }
        if (result != Integer.MAX_VALUE) {
            return s.substring(head, head + result);
        }
        return "";
    }


    // --dfs优先遍历----- //

    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0) {
            return false;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && intervalExist(i, j, 0, board, word, used)) {
                    return true;
                }
            }
        }
        return false;

    }

    private boolean intervalExist(int i, int j, int k, char[][] board, String word, boolean[][] used) {
        if (k == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || used[i][j] || board[i][j] != word.charAt(k)) {
            return false;
        }
        used[i][j] = true;
        if (intervalExist(i - 1, j, k + 1, board, word, used) ||
                intervalExist(i + 1, j, k + 1, board, word, used) ||
                intervalExist(i, j - 1, k + 1, board, word, used) ||
                intervalExist(i, j + 1, k + 1, board, word, used)) {
            return true;
        }
        used[i][j] = false;

        return false;
    }


}