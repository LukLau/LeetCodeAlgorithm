package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.Interval;
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
        String s = "3+2*2";

        System.out.println(solution.calculateII(s));

    }

    // ---O log(N)算法---- //

    //计算最大质因数
    public static int getTheLargestPrimeFactor(int n) {
        int returnFactor = 1;
        for (int factor = 2; n > 1; factor++) {
            if (n % factor == 0) {
                n = n / factor;
                returnFactor = factor;
                while (n % factor == 0) {
                    n = n / factor;
                }
            }
        }
        return returnFactor;
    }

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


    // --查找数组中的最小值---- //

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
     * 153. Find Minimum in Rotated Sorted Array
     *
     * @param nums
     * @return
     */
    public int findMin(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= nums[right]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return nums[left];
    }

    /**
     * 154. Find Minimum in Rotated Sorted Array II
     *
     * @param nums
     * @return
     */
    public int findMinWithRepeat(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;

            if (nums[mid] < nums[right]) {
                right = mid;
            } else if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right--;
            }
        }
        return nums[left];
    }


    // --正则表达式匹配问题 //

    /**
     * 162. Find Peak Element
     *
     * @param nums
     * @return
     */
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
        return nums[left];
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


    // --旋转数组系列问题-- //

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

    // ---排列组合问题---- //

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

    /**
     * 90. Subsets II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        intervalSubsetsWithDup(result, new ArrayList<>(), 0, nums);
        return result;
    }

    private void intervalSubsetsWithDup(List<List<Integer>> result, List<Integer> tmp, int start, int[] nums) {
        result.add(new ArrayList<>(tmp));
        for (int i = start; i < nums.length; i++) {
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            tmp.add(nums[i]);
            intervalSubsetsWithDup(result, tmp, i + 1, nums);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 216. Combination Sum III
     *
     * @param k
     * @param n
     * @return
     */
    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> result = new ArrayList<>();
        intervalCombinationSum3(result, new ArrayList<>(), 1, k, n);
        return result;
    }

    private void intervalCombinationSum3(List<List<Integer>> result, ArrayList<Integer> tmp, int start, int k, int value) {
        if (tmp.size() == k && value == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= 9 && i <= value; i++) {
            tmp.add(i);
            intervalCombinationSum3(result, tmp, i + 1, k, value - i);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * todo
     * 254 Factor Combinations
     *
     * @param n: a integer
     * @return: return a 2D array
     */
    public List<List<Integer>> getFactors(int n) {
        // write your code here
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        intervalGetFactors(result, new ArrayList<Integer>(), 2, n / 2, 1, n);
        return result;
    }

    private void intervalGetFactors(List<List<Integer>> result, ArrayList<Integer> tmp, int start, int end, int value, int n) {
        if (value == n) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        if (value > n) {
            return;
        }
        for (int i = start; i <= end && (i * value <= n); i++) {
            if (n % i != 0) {
                continue;
            }
            tmp.add(i);
            intervalGetFactors(result, tmp, i, end, i * value, n);
            tmp.remove(tmp.size() - 1);
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
        if (s == null) {
            return false;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return false;
        }
        boolean seenNumber = false;
        boolean seenE = false;
        boolean seenAfterE = true;
        boolean seenDigit = false;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            char word = words[i];
            if (Character.isDigit(word)) {
                seenAfterE = true;
                seenNumber = true;
            } else if (word == 'e' || word == 'E') {
                if (seenE || i == 0) {
                    return false;
                }
                if (!seenNumber) {
                    return false;
                }
                if (!Character.isDigit(words[i - 1]) && words[i - 1] != '.') {
                    return false;
                }
                seenE = true;
                seenAfterE = false;
            } else if (word == '.') {
                // if (i != 0 && !Character.isDigit(words[i - 1])) {
                //     return false;
                // }
                if (seenDigit || seenE) {
                    return false;
                }
                seenDigit = true;
            } else if (word == '-' || word == '+') {
                if (i != 0 && (words[i - 1] != 'e' && words[i - 1] != 'E')) {
                    return false;
                }
            } else {
                return false;
            }

        }
        return seenAfterE && seenNumber;
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
     * 92. Reverse Linked List II
     *
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null) {
            return null;
        }
        ListNode root = new ListNode(0);
        root.next = head;
        ListNode dummy = root;
        for (int i = 0; i < m - 1; i++) {
            dummy = dummy.next;
        }
        ListNode start = dummy.next;

        ListNode then = start.next;

        for (int i = 0; i < n - m; i++) {
            start.next = then.next;

            then.next = dummy.next;

            dummy.next = then;

            then = start.next;
        }
        return root.next;
    }


    /**
     * 237. Delete Node in a Linked List
     *
     * @param node
     */
    public void deleteNode(ListNode node) {
        if (node.next.next == null) {
            node.val = node.next.val;
            node.next = null;
        } else {
            node.val = node.next.val;
            node.next = node.next.next;
        }

    }


    // ----- //

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

    public boolean searchMatrixII(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int i = row - 1;
        int j = 0;
        while (i >= 0 && j < column) {
            int val = matrix[i][j];
            if (val == target) {
                return true;
            } else if (val < target) {
                j++;
            } else {
                i--;
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


    /**
     * 239. Sliding Window Maximum
     *
     * @param nums
     * @param k
     * @return
     */
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0) {
            return new int[] {};
        }
        List<Integer> result = new ArrayList<>();
        LinkedList<Integer> linkedList = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            int index = i - k + 1;
            if (!linkedList.isEmpty() && linkedList.peek() < index) {
                linkedList.poll();
            }
            while (!linkedList.isEmpty() && nums[linkedList.getLast()] <= nums[i]) {
                linkedList.pollLast();
            }
            linkedList.offer(i);
            if (index >= 0) {
                result.add(nums[linkedList.peek()]);
            }
        }
        int[] tmp = new int[result.size()];
        for (int i = 0; i < tmp.length; i++) {
            tmp[i] = result.get(i);
        }
        return tmp;
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


    // --唯一路径问题-- //


    /**
     * 62. Unique Paths
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        if (m < 0 || n < 0) {
            return 0;
        }
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[j] = j == 0 ? dp[j] : dp[j] + dp[j - 1];
            }
        }
        return dp[n - 1];
    }


    /**
     * 63. Unique Paths II
     *
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int row = obstacleGrid.length;
        int column = obstacleGrid[0].length;
        int[] dp = new int[column];

        dp[0] = 1;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else {
                    dp[j] = j == 0 ? dp[j] : dp[j - 1] + dp[j];
                }
            }
        }
        return dp[column - 1];
    }


    /**
     * 64. Minimum Path Sum
     *
     * @param grid
     * @return
     */
    public int minPathSumV2(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int[] dp = new int[column];
        dp[0] = grid[0][0];
        for (int j = 1; j < column; j++) {
            dp[j] = dp[j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                dp[j] = grid[i][j] + (j == 0 ? dp[j] : Math.min(dp[j - 1], dp[j]));
            }
        }
        return dp[column - 1];
    }

    // --移除重复元素链表 - //

    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val == head.next.val) {
            ListNode current = head.next;
            while (current != null && current.val == head.val) {
                current = current.next;
            }
            return deleteDuplicates(current);
        }
        head.next = deleteDuplicates(head.next);
        return head;
    }

    public int largestRectangleArea(int[] heights) {
        if (heights == null || heights.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();

        int result = 0;

        for (int i = 0; i <= heights.length; i++) {
            int height = i == heights.length ? 0 : heights[i];
            if (stack.isEmpty() || heights[stack.peek()] < height) {
                stack.push(i);
            } else {
                int rightEdge = heights[stack.pop()];
                int side = stack.isEmpty() ? i : i - stack.peek() - 1;
                result = Math.max(result, rightEdge * side);
                i--;
            }
        }
        return result;
    }

    //--动态规划问题---- //

    /**
     * 85. Maximal Rectangle
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int result = 0;

        int row = matrix.length;

        int column = matrix[0].length;

        int[][] dp = new int[row][column];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '0') {
                    dp[i][j] = -1;
                    continue;
                }
                dp[i][j] = j;
                if (j != 0 && dp[i][j - 1] >= 0) {
                    dp[i][j] = dp[i][j - 1];
                }
                int width = dp[i][j];
                for (int k = i; k >= 0 && matrix[k][j] == '1'; k--) {
                    width = Math.max(width, dp[k][j]);
                    result = Math.max(result, (j - width + 1) * (i - k + 1));
                }
            }
        }
        return result;
    }

    public int maximalRectangleV2(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;
        int column = matrix[0].length;
        int[] height = new int[column];
        int[] left = new int[column];
        int[] right = new int[column];
        Arrays.fill(right, column);
        int result = 0;
        for (int i = 0; i < row; i++) {
            int leftSide = 0;
            int rightSide = column;
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '0') {
                    height[j] = 0;
                    left[j] = 0;
                    leftSide = j + 1;
                } else {
                    height[j]++;
                    left[j] = Math.max(leftSide, left[j]);
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '0') {
                    right[j] = column;
                    rightSide = j;
                } else {
                    right[j] = Math.min(right[j], rightSide);
                }
            }
            for (int j = 0; j < column; j++) {
                if (height[j] == 0) {
                    continue;
                }
                result = Math.max(result, (right[j] - left[j]) * height[j]);
            }
        }
        return result;
    }


    // ---格雷码---- //


    /**
     * todo
     * 89. Gray Code
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        List<Integer> result = new ArrayList<>();
        return null;
    }


    /**
     * 142. Linked List Cycle II
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return slow;
            }
        }
        return null;
    }


    /**
     * 143. Reorder List
     *
     * @param head
     */
    public void reorderList(ListNode head) {
        if (head == null || head.next == null) {
            return;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        ListNode reverseList = reverseList(slow.next);

        slow.next = null;

        slow = head;

        while (slow != null && reverseList != null) {
            ListNode tmp = slow.next;

            ListNode reverseListTmp = reverseList.next;

            slow.next = reverseList;

            reverseList.next = tmp;

            slow = tmp;

            reverseList = reverseListTmp;
        }
    }

    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        while (head != null) {
            ListNode node = head.next;
            head.next = prev;
            prev = head;
            head = node;
        }
        return prev;
    }


    // ---逆波兰系列--- //

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
        Stack<Integer> stack = new Stack<>();
        List<String> sign = Arrays.asList("+", "-", "*", "/");
        for (String token : tokens) {
            if (sign.contains(token)) {
                Integer firstNum = stack.pop();
                Integer secondNum = stack.pop();
                if ("+".equals(token)) {
                    stack.push(firstNum + secondNum);
                } else if ("-".equals(token)) {
                    stack.push(secondNum - firstNum);
                } else if ("*".equals(token)) {
                    stack.push(firstNum * secondNum);
                } else {
                    stack.push(secondNum / firstNum);
                }
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    // ---计算器系列--- //


    /**
     * todo
     * 224. Basic Calculator
     *
     * @param s
     * @return
     */
    public int calculate(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        int sign = 1;
        char[] words = s.toCharArray();
        Stack<Integer> stack = new Stack<>();
        int end = 0;
        while (end < words.length) {
            char word = words[end];
            if (Character.isDigit(word)) {
                int tmp = 0;
                while (end < words.length && Character.isDigit(words[end])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[end]);
                    end++;
                }
                result += sign * tmp;
            } else {
                if (word == '+') {
                    sign = 1;
                } else if (word == '-') {
                    sign = -1;
                } else if (word == '(') {
                    stack.push(result);
                    stack.push(sign);
                    result = 0;
                    sign = 1;
                } else if (word == ')') {
                    result = stack.pop() * result + stack.pop();
                }
            }
        }
        return result;
    }


    /**
     * todo
     * 227. Basic Calculator II
     *
     * @param s
     * @return
     */
    public int calculateII(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        char[] words = s.toCharArray();
        int end = 0;
        int tmp = 0;
        int len = words.length;
        while (end < len) {
            if (Character.isDigit(words[end])) {
                while (end < len && Character.isDigit(words[end])) {
                    tmp = tmp * 10 + Character.getNumericValue(words[end]);
                    end++;
                }
            }
            boolean lastIndex = end == len;
            if (lastIndex || (!Character.isDigit(words[end]) && words[end] != ' ')) {
                if (sign == '+') {
                    stack.push(tmp);
                } else if (sign == '-') {
                    stack.push(-tmp);
                } else if (sign == '*') {
                    stack.push(stack.pop() * tmp);
                } else if (sign == '/') {
                    stack.push(stack.pop() / tmp);
                }
            }
            if (end != len && words[end] != ' ') {
                sign = words[end];
                tmp = 0;
            }
            end++;
        }
        int result = 0;
        for (Integer num : stack) {
            result += num;
        }
        return result;
    }

    // --- 单词最短距离系列 ---//


    /**
     * 243 Shortest Word Distance
     *
     * @param words: a list of words
     * @param word1: a string
     * @param word2: a string
     * @return: the shortest distance between word1 and word2 in the list
     */
    public int shortestDistance(String[] words, String word1, String word2) {
        // Write your code here
        int result = Integer.MAX_VALUE;
        int leftIndex = -1;
        int rightIndex = -1;
        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            if (word.equals(word1)) {
                leftIndex = i;
            } else if (word.equals(word2)) {
                rightIndex = i;
            }
            if (leftIndex != -1 && rightIndex != -1) {
                result = Math.min(result, Math.abs(leftIndex - rightIndex));
            }
        }
        return result;
    }


    // ---会议室问题---//

    /**
     * 252 Meeting Rooms
     *
     * @param intervals: an array of meeting time intervals
     * @return: if a person could attend all meetings
     */
    public boolean canAttendMeetings(List<Interval> intervals) {
        if (intervals == null) {
            return false;
        }
        int size = intervals.size();
        if (size <= 1) {
            return true;
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));

        int endTime = intervals.get(0).end;
        for (int i = 1; i < size; i++) {
            Interval current = intervals.get(i);
            if (endTime <= current.start) {
                endTime = current.end;
            } else {
                return false;
            }
        }
        return true;
        // Write your code here
    }


    /**
     * todo
     * 253 Meeting Rooms II
     *
     * @param intervals: an array of meeting time intervals
     * @return: the minimum number of conference rooms required
     */
    public int minMeetingRooms(List<Interval> intervals) {
        if (intervals == null) {
            return 0;
        }
        int size = intervals.size();
        if (size <= 1) {
            return size;
        }
        intervals.sort(Comparator.comparingInt(o -> o.start));
        PriorityQueue<Interval> priorityQueue = new PriorityQueue<>(Comparator.comparingInt(o -> o.end));

        priorityQueue.offer(intervals.get(0));
        for (int i = 1; i < size; i++) {
            Interval poll = priorityQueue.poll();
            Interval current = intervals.get(i);
            if (current.start >= poll.end) {
                poll.end = current.end;
            } else {
                priorityQueue.offer(current);
            }
            priorityQueue.offer(poll);
        }
        return priorityQueue.size();


        // Write your code here
    }

    // --- 画房子问题---//

    /**
     * todo
     * 256 Paint House
     *
     * @param costs: n x 3 cost matrix
     * @return: An integer, the minimum cost to paint all houses
     */
    public int minCost(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int result = Integer.MAX_VALUE;

        int column = costs[0].length;

        int row = costs.length;
        int[][] dp = new int[row][column];
        System.arraycopy(costs[0], 0, dp[0], 0, costs[0].length);
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[i][j] = Math.min(dp[i - 1][1], dp[i - 1][2]) + costs[i][j];
                } else {
                    dp[i][j] = Math.min(dp[i - 1][(j + 1) % column], dp[i - 1][(j + 2) % column]) + costs[i][j];
                }
            }
        }
        for (int j = 0; j < column; j++) {
            result = Math.min(result, dp[row - 1][j]);
        }
        return result;
        // write your code here
    }


    public int minCostV2(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        int column = costs[0].length;

        for (int i = 1; i < costs.length; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
        }
        return Math.min(Math.min(costs[row - 1][0], costs[row - 1][1]), costs[row - 1][2]);
        // write your code here
    }


    /**
     * todo
     * 265
     * Paint House II
     *
     * @param costs: n x k cost matrix
     * @return: an integer, the minimum cost to paint all houses
     */
    public int minCostII(int[][] costs) {
        // write your code here
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int row = costs.length;
        int column = costs[0].length;

        int firstSmallIndex = -1;
        int secondSmallIndex = -1;
        for (int i = 0; i < row; i++) {
            int lastFirstIndex = firstSmallIndex;
            int lastSecondIndex = secondSmallIndex;
            firstSmallIndex = -1;
            secondSmallIndex = -1;
            for (int j = 0; j < column; j++) {

                if (lastFirstIndex >= 0) {
                    if (j != lastFirstIndex) {
                        costs[i][j] += costs[i - 1][lastFirstIndex];
                    } else {
                        costs[i][j] += costs[i - 1][lastSecondIndex];
                    }
                }

                if (firstSmallIndex < 0 || costs[i][j] < costs[i][firstSmallIndex]) {
                    secondSmallIndex = firstSmallIndex;
                    firstSmallIndex = j;
                } else if (secondSmallIndex < 0 || costs[i][j] < costs[i][secondSmallIndex]) {
                    secondSmallIndex = j;
                }
            }
        }
        return costs[row - 1][firstSmallIndex];
    }


    /**
     * 优化上面空间
     *
     * @param costs
     * @return
     */
    public int minCostIIV2(int[][] costs) {
        if (costs == null || costs.length == 0) {
            return 0;
        }
        int column = costs[0].length;
        int row = costs.length;
        int minValue1 = -1;
        int minValue2 = -1;
        int idx = -1;
        for (int i = 0; i < row; i++) {


            for (int j = 0; j < column; j++) {

//                costs[i][j] += (j == idx ? costs[i-1][j] : costs[]);


            }
        }
        return -1;

    }


    /**
     * 266
     * Palindrome Permutation
     *
     * @param s: the given string
     * @return: if a permutation of the string could form a palindrome
     */
    public boolean canPermutePalindrome(String s) {
        if (s == null) {
            return false;
        }
        int len = s.length();
        if (len <= 0) {
            return true;
        }
        boolean occurOdd = false;
        int[] hash = new int[256];
        for (char word : s.toCharArray()) {
            hash[word]++;
        }
        for (int num : hash) {
            if (num % 2 != 0) {
                if (occurOdd) {
                    return false;
                }
                occurOdd = true;
            }
        }
        return true;
        // write your code here
    }


}
