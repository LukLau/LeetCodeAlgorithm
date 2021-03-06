package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.TreeNode;

import java.util.*;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/6/25
 */
public class FirstPage {

    private int beginPoint = 0;
    private int longestLen = Integer.MIN_VALUE;

    public static void main(String[] args) {
        FirstPage page = new FirstPage();
        String[] words = new String[]{"This", "is", "an", "example", "of", "text", "justification."};
        page.restoreIpAddresses("9245587303");
    }

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
        int[] result = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                result[0] = map.get(target - nums[i]);
                result[1] = i;
                return result;
            }
            map.put(nums[i], i);
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
        ListNode root = new ListNode(0);
        ListNode dummy = root;
        int carry = 0;
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
     * 6. ZigZag Conversion
     *
     * @param s
     * @param numRows
     * @return
     */
    public String convert(String s, int numRows) {
        if (s == null || s.isEmpty()) {
            return "";
        }
        StringBuilder[] builders = new StringBuilder[numRows];
        for (int i = 0; i < builders.length; i++) {
            builders[i] = new StringBuilder();
        }
        char[] chars = s.toCharArray();
        int index = 0;
        while (index < chars.length) {
            for (int i = 0; i < numRows && index < chars.length; i++) {
                builders[i].append(chars[index++]);
            }
            for (int i = numRows - 2; i >= 1 && index < chars.length; i--) {
                builders[i].append(chars[index++]);
            }
        }
        for (int i = 1; i < builders.length; i++) {
            builders[0].append(builders[i]);
        }
        return builders[0].toString();
    }


    /**
     * 7. Reverse Integer
     *
     * @param x
     * @return
     */
    public int reverse(int x) {
        if (x == 0) {
            return 0;
        }
        long result = 0;
        while (x != 0) {
            result = result * 10 + x % 10;

            x /= 10;

            if (result > Integer.MAX_VALUE || result < Integer.MIN_VALUE) {
                return 0;
            }

        }
        return (int) result;
    }

    /**
     * 8. 字符串转换整数 (atoi)
     *
     * @param s
     * @return
     */
    public int myAtoi(String s) {
        if (s == null) {
            return 0;
        }
        s = s.trim();
        if (s.isEmpty()) {
            return 0;
        }
        int sign = 1;
        int index = 0;
        char[] words = s.toCharArray();
        char firstWord = words[index];
        if (firstWord == '-' || firstWord == '+') {
            sign = firstWord == '-' ? -1 : 1;
            index++;
        }
        long result = 0;
        while (index < words.length && Character.isDigit(words[index])) {
            int val = Character.getNumericValue(words[index]);
            result = result * 10 + val;

            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            index++;
        }
        return (int) result * sign;
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
        for (int j = 1; j <= n; j++) {
            if (p.charAt(j - 1) == '*') {
                dp[0][j] = dp[0][j - 2];
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') {
                    dp[i][j] = dp[i - 1][j - 1];
                } else if (p.charAt(j - 1) == '*') {
                    if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
                        dp[i][j] = dp[i][j - 2];
                    } else {
                        dp[i][j] = dp[i - 1][j] || dp[i][j - 2] || dp[i][j - 1];
                    }
                }
            }
        }
        return dp[m][n];
    }

    /**
     * todo 递归解法
     *
     * @param s
     * @param p
     * @return
     */
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

    /**
     * 11. 盛最多水的容器
     *
     * @param height
     * @return
     */
    public int maxArea(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            result = Math.max(result, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return result;
    }

    /**
     * 12. 整数转罗马数字
     *
     * @param num
     * @return
     */
    public String intToRoman(int num) {
        if (num <= 0) {
            return "";
        }
        String[] one = new String[]{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"};
        String[] two = new String[]{"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"};
        String[] three = new String[]{"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"};
        String[] four = new String[]{"", "M", "MM", "MMM"};

        StringBuilder builder = new StringBuilder();

        return builder.append(four[num / 1000])
                .append(three[(num / 100) % 10])
                .append(two[(num / 10) % 10])
                .append(one[num % 10]).toString();
    }

    /**
     * 13. 罗马数字转整数
     *
     * @param s
     * @return
     */
    public int romanToInt(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<>();

        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);

        int result = 0;

        char[] chars = s.toCharArray();

        for (char word : chars) {
            Integer val = map.get(word);
            result += val;
        }
        for (int i = 1; i < chars.length; i++) {

        }

        return result;
    }

    /**
     * 14. 最长公共前缀
     *
     * @param strs
     * @return
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
            }
        }
        return prefix;
    }

    /**
     * 15. 三数之和
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int val = nums[i] + nums[left] + nums[right];
                if (val == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;
                } else if (val < 0) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return result;
    }


    /**
     * 16. 最接近的三数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(int[] nums, int target) {

        int ans = nums[0] + nums[1] + nums[2];

        Arrays.sort(nums);

        int len = nums.length;
        for (int i = 0; i < len - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = len - 1;

            while (left < right) {
                int value = nums[i] + nums[left] + nums[right];

                if (value == target) {
                    return target;
                }
                if (Math.abs(ans - target) > Math.abs(value - target)) {
                    ans = value;
                }
                if (value < target) {
                    left++;
                } else {
                    right--;
                }
            }
        }
        return ans;
    }

    /**
     * 17. 电话号码的字母组合
     *
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits == null || digits.isEmpty()) {
            return new ArrayList<>();
        }
        String[] nums = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        LinkedList<String> linkedList = new LinkedList<>();
        linkedList.add("");
        int len = digits.length();
        for (int i = 0; i < len; i++) {
            int index = Character.getNumericValue(digits.charAt(i));
            String word = nums[index];
            while (linkedList.peek().length() == i) {
                String poll = linkedList.poll();
                char[] words = word.toCharArray();
                for (char tmp : words) {
                    linkedList.offer(poll + tmp);
                }
            }
        }
        return linkedList;
    }

    /**
     * 19. 删除链表的倒数第N个节点
     *
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return null;
        }
//        ListNode root = new ListNode(0);
//        root.next = head;
        int count = 0;
        ListNode node = head;
        while (node != null) {
            count++;

            node = node.next;
        }
        count++;

        ListNode root = new ListNode(0);

        root.next = head;

        ListNode fast = root;


        for (int i = 0; i < count - n - 1; i++) {
            fast = fast.next;
        }
        fast.next = fast.next.next;

        return root.next;

    }

    /**
     * 20. 有效的括号
     *
     * @param s
     * @return
     */
    public boolean isValid(String s) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        Stack<Character> stack = new Stack<>();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (word == '{') {
                stack.push('}');
            } else if (word == '[') {
                stack.push(']');
            } else if (word == '(') {
                stack.push(')');
            } else {
                if (stack.isEmpty()) {
                    return false;
                }
                if (stack.peek().equals(word)) {
                    stack.pop();
                } else {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. 合并两个有序链表
     *
     * @param l1
     * @param l2
     * @return
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        if (l1.val <= l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }


    /**
     * 22. 括号生成
     *
     * @param n
     * @return
     */
    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalGenerateParenthesis(result, "", 0, 0, n);
        return result;
    }

    private void intervalGenerateParenthesis(List<String> result, String s, int open, int close, int n) {
        if (s.length() == 2 * n) {
            result.add(s);
            return;
        }
        if (open < n) {
            intervalGenerateParenthesis(result, s + "(", open + 1, close, n);
        }
        if (close < open) {
            intervalGenerateParenthesis(result, s + ")", open, close + 1, n);
        }
    }

    /**
     * 23. 合并K个排序链表
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

    /**
     * 24. 两两交换链表中的节点
     *
     * @param head
     * @return
     */
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);
        root.next = head;


        ListNode slow = null;

        ListNode fast = null;


        ListNode dummy = root;
        while (dummy.next != null && dummy.next.next != null) {
            fast = dummy.next.next;
            slow = dummy.next;

            slow.next = fast.next;

            fast.next = slow;

            dummy.next = fast;

            dummy = dummy.next.next;
        }
        return root.next;
    }

    /**
     * 25. K 个一组翻转链表
     *
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || head.next == null || k <= 0) {
            return null;
        }
        ListNode node = head;
        int count = 0;
        while (node.next != null && count != k) {
            node = node.next;
            count++;
        }
        if (count == k) {
            ListNode reverseNode = reverseKGroup(node, k);
            while (count-- > 0) {
                ListNode tmp = head.next;

                head.next = reverseNode;

                reverseNode = head;

                head = tmp;
            }
            head = reverseNode;
        }
        return head;
    }

    public ListNode reverseKGroupV2(ListNode head, int k) {
        if (head == null || head.next == null || k <= 0) {
            return head;
        }
        ListNode node = head;
        for (int i = 0; i < k; i++) {
            if (node == null) {
                return head;
            }
            node = node.next;
        }
        ListNode reverseHead = reverseList(head, node);

        head.next = reverseKGroupV2(node, k);

        return reverseHead;
    }

    private ListNode reverseList(ListNode start, ListNode end) {
        ListNode prev = end;
        while (start != end) {
            ListNode tmp = start.next;
            start.next = prev;
            prev = start;
            start = tmp;
        }
        return prev;
    }

    /**
     * 26. 删除排序数组中的重复项
     *
     * @param nums
     * @return
     */
    public int removeDuplicates(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[index++] = nums[i];
            }
        }
        return index;
    }

    /**
     * 29. 两数相除
     *
     * @param dividend
     * @param divisor
     * @return
     */
    public int divide(int dividend, int divisor) {
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        if (dividend == 0) {
            return 0;
        }
        int sign = ((dividend < 0 && divisor < 0) || (dividend > 0 && divisor > 0)) ? 1 : -1;

        long dvd = Math.abs((long) dividend);

        long dvs = Math.abs((long) divisor);

        int result = 0;
        while (dvd >= dvs) {
            long tmp = dvs;
            int number = 1;
            while (dvd >= (tmp << 1)) {
                tmp <<= 1;
                number <<= 1;
            }
            dvd -= tmp;
            result += number;
        }
        return sign * result;

    }

    /**
     * 30. 串联所有单词的子串
     *
     * @param s
     * @param words
     * @return
     */
    public List<Integer> findSubstring(String s, String[] words) {
        return null;
    }

    /**
     * 31. 下一个排列
     *
     * @param nums
     */
    public void nextPermutation(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int length = nums.length - 1;

        int index = length;

        while (index >= 1) {
            if (nums[index] > nums[index - 1]) {
                break;
            }
            index--;
        }
        if (index == 0) {
            reverseNums(nums, 0, length - 1);
        } else {
            int val = nums[index - 1];
            int j = length;
            while (j > index - 1) {
                if (nums[j] > val) {
                    break;
                }
                j--;
            }
            swap(nums, j, index - 1);
            reverseNums(nums, index, length);
        }
    }

    private void swap(int[] nums, int i, int j) {
        if (i == j) {
            return;
        }
        int val = nums[i];
        nums[i] = nums[j];
        nums[j] = val;
    }

    private void reverseNums(int[] nums, int start, int end) {
        if (start > end) {
            return;
        }
        int mid = (start + end) / 2;
        for (int i = start; i <= mid; i++) {
            int val = nums[i];
            nums[i] = nums[start + end - i];
            nums[start + end - i] = val;
        }
    }

    /**
     * 32. 最长有效括号
     *
     * @param s
     * @return
     */
    public int longestValidParentheses(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int result = 0;
        Stack<Integer> stack = new Stack<>();
        int left = -1;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            if (words[i] == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && words[stack.peek()] == '(') {
                    stack.pop();
                    if (stack.isEmpty()) {
                        result = Math.max(result, i - left);
                    } else {
                        result = Math.max(result, i - stack.peek());
                    }
                } else {
                    left = i;
                }
            }
        }
        return result;
    }

    public int longestValidParenthesesV2(String s) {
        if (s == null) {
            return 0;
        }
        if (s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (stack.isEmpty() || word == '(') {
                stack.push(i);
            } else {
                if (s.charAt(stack.peek()) == '(') {
                    stack.pop();
                } else {
                    stack.push(i);
                }
            }
        }
        if (stack.isEmpty()) {
            return len;
        }
        int result = 0;
        int a = s.length();
        while (!stack.isEmpty()) {
            int m = stack.pop();
            result = Math.max(result, a - m - 1);
            a = m;
        }
        result = Math.max(result, a);
        return result;
    }

    public int[] searchRangeV2(int[] nums, int target) {
        int[] result = new int[]{-1, -1};
        if (nums == null || nums.length == 0) {
            return result;
        }
        int leftIndex = findLeftTargetIndex(nums, 0, nums.length - 1, target);

        int rightIndex = findRightTargetIndex(nums, 0, nums.length - 1, target);

        if (leftIndex == -1 || rightIndex == -1) {
            return result;
        }
        result[0] = leftIndex;

        result[1] = rightIndex;

        return result;

    }

    private int findRightTargetIndex(int[] nums, int left, int right, int target) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else {
                if (mid < nums.length - 1 && nums[mid] == nums[mid + 1]) {
                    left = mid + 1;
                } else {
                    return mid;
                }
            }
        }
        return -1;
    }

    private int findLeftTargetIndex(int[] nums, int left, int right, int target) {
        if (left > right) {
            return -1;
        }
        int mid = left + (right - left) / 2;

        if (nums[mid] < target) {
            return findLeftTargetIndex(nums, mid + 1, right, target);
        } else if (nums[mid] > target) {
            return findLeftTargetIndex(nums, left, mid - 1, target);
        } else {
            if (mid > 0 && nums[mid - 1] == target) {
                return findLeftTargetIndex(nums, left, mid - 1, target);
            }
            return mid;
        }
    }


    /**
     * todo
     * 35. 搜索插入位置
     *
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            int val = nums[mid];
            if (val == target) {
                return mid;
            } else if (val < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    /**
     * 39. 组合总和
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<>();
        intervalCombination(result, new ArrayList<>(), 0, candidates, target);
        return result;
    }

    private void intervalCombination(List<List<Integer>> result,
                                     ArrayList<Integer> integers,
                                     int start,
                                     int[] candidates,
                                     int target) {
        if (target == 0) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            integers.add(candidates[i]);
            intervalCombination(result, integers, i, candidates, target - candidates[i]);
            integers.remove(integers.size() - 1);
        }
    }

    /**
     * 40. Combination Sum II
     *
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates == null || candidates.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(candidates);
        List<List<Integer>> result = new ArrayList<>();
        intervalCombinationSums(result, new ArrayList<>(), 0, candidates, target);
        return result;
    }

    private void intervalCombinationSums(List<List<Integer>> result, ArrayList<Integer> tmp, int start, int[] candidates, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            intervalCombinationSums(result, tmp, i + 1, candidates, target - candidates[i]);
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 41. First Missing Positive
     *
     * @param nums
     * @return
     */
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] >= 1 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
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

    /**
     * 42. Trapping Rain Water
     *
     * @param height
     * @return
     */
    public int trap(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }
        int result = 0;
        int left = 0;
        int right = height.length - 1;
        int leftSide = 0;
        int rightSide = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                if (height[left] >= leftSide) {
                    leftSide = height[left];
                } else {
                    result += leftSide - height[left];
                }
                left++;
            } else {
                if (height[right] >= rightSide) {
                    rightSide = height[right];
                } else {
                    result += rightSide - height[right];
                }
                right--;
            }
        }
        return result;
    }

    /**
     * @param height
     * @return
     */
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
            int minEdge = Math.min(height[left], height[right]);

            for (int i = left; i <= right; i++) {
                int val = height[i];

                if (val < minEdge) {
                    result += minEdge - height[i];
                    height[i] = 0;
                } else {
                    height[i] -= minEdge;
                }
            }
        }
        return result;
    }

    /**
     * 43. Multiply Strings
     *
     * @param num1
     * @param num2
     * @return
     */
    public String multiply(String num1, String num2) {
        if (num1.isEmpty() && num2.isEmpty()) {
            return "0";
        }
        int m = num1.length();
        int n = num2.length();
        int[] position = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            for (int j = n - 1; j >= 0; j--) {
                int val = Character.getNumericValue(num1.charAt(i)) * Character.getNumericValue(num2.charAt(j)) + position[i + j + 1];

                position[i + j + 1] = val % 10;

                position[i + j] += val / 10;
            }
        }
        StringBuilder builder = new StringBuilder();
        for (int num : position) {

            if (!(builder.length() == 0 && num == 0)) {
                builder.append(num);
            }
        }
        return builder.length() == 0 ? "0" : builder.toString();
    }


    /**
     * 45. Jump Game II
     *
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        int step = 0;
        int furthest = nums[0];
        int current = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            furthest = Math.max(nums[i] + i, furthest);
            if (i == current) {
                step++;
                current = furthest;
            }
        }
        return step;

    }

    /**
     * 46. Permutations
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permute(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        intervalPermute(result, new ArrayList<>(), nums, used);
        return result;
    }

    private void intervalPermute(List<List<Integer>> result, ArrayList<Integer> tmp, int[] nums, boolean[] used) {
        if (tmp.size() == nums.length) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            used[i] = true;
            tmp.add(nums[i]);
            intervalPermute(result, tmp, nums, used);
            used[i] = false;
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 47. Permutations II
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        List<List<Integer>> result = new ArrayList<>();
        intervalPermuteUnique(result, new ArrayList<>(), used, nums);
        return result;
    }

    private void intervalPermuteUnique(List<List<Integer>> result, ArrayList<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            result.add(new ArrayList<>(integers));
        }
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            if (used[i]) {
                continue;
            }

            used[i] = true;
            integers.add(nums[i]);
            intervalPermuteUnique(result, integers, used, nums);
            used[i] = false;
            integers.remove(integers.size() - 1);
        }

    }


    /**
     * 48. Rotate Image
     *
     * @param matrix
     */
    public void rotate(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < i; j++) {
                swapMatrix(matrix, i, j);
            }
        }
        for (int[] row : matrix) {
            int start = 0;
            int end = row.length - 1;
            for (int i = start; i <= (start + end) / 2; i++) {
                swap(row, i, start + end - i);
            }
        }

    }

    private void swapMatrix(int[][] matrix, int i, int j) {
        int val = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = val;
    }

    /**
     * 49. Group Anagrams
     *
     * @param strs
     * @return
     */
    public List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        Map<String, List<String>> map = new HashMap<>();

        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);

            List<String> list = map.getOrDefault(key, new ArrayList<>());

            list.add(str);

            map.put(key, list);
        }
        return new ArrayList<>(map.values());
    }


    /**
     * 50. Pow(x, n)
     *
     * @param x
     * @param n
     * @return
     */
    public double myPow(double x, int n) {
        if (n == 1) {
            return x;
        }
        if (n == 0) {
            return 1;
        }
        if (n < 0) {
            x = 1 / x;
            n = -n;
        }
        return n % 2 == 0 ? myPow(x * x, n / 2) : x * myPow(x * x, n / 2);
    }


    public double myPowV2(double x, int n) {
        long sign = Math.abs((long) n);
        double result = 1.0;
        while (sign != 0) {
            if (sign % 2 != 0) {
                result *= x;
            }
            if (result > Integer.MAX_VALUE || result < Integer.MIN_VALUE) {
                return 0;
            }
            x *= x;
            sign >>= 1;
        }
        return n < 0 ? 1 / result : result;
    }


    /**
     * 51. N-Queens
     *
     * @param n
     * @return
     */
    public List<List<String>> solveNQueens(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        char[][] queens = new char[n][n];
        for (char[] row : queens) {
            Arrays.fill(row, '.');
        }
        List<List<String>> result = new ArrayList<>();
        intervalSolveNQueens(result, 0, queens);
        return result;
    }

    private void intervalSolveNQueens(List<List<String>> result, int row, char[][] matrix) {
        if (row == matrix.length) {
            List<String> tmp = new ArrayList<>();
            for (char[] content : matrix) {
                tmp.add(String.valueOf(content));
            }
            result.add(tmp);
            return;
        }
        for (int i = 0; i < matrix[0].length; i++) {
            if (validNQueens(i, row, matrix)) {
                matrix[row][i] = 'Q';
                intervalSolveNQueens(result, row + 1, matrix);
                matrix[row][i] = '.';
            }
        }
    }

    private boolean validNQueens(int column, int row, char[][] matrix) {
        for (int i = row - 1; i >= 0; i--) {
            if (matrix[i][column] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column - 1; i >= 0 && j >= 0; i--, j--) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        for (int i = row - 1, j = column + 1; i >= 0 && j < matrix[i].length; i--, j++) {
            if (matrix[i][j] == 'Q') {
                return false;
            }
        }
        return true;
    }

    /**
     * 52. N-Queens II
     *
     * @param n
     * @return
     */
    public int totalNQueens(int n) {
        if (n <= 0) {
            return 0;
        }

        int[] dp = new int[n];

        return intervalTotalNQueens(dp, 0, n);
    }

    private int intervalTotalNQueens(int[] dp, int row, int n) {
        int count = 0;
        if (row == n) {
            count++;
            return count;
        }
        for (int j = 0; j < n; j++) {
            if (validNQueens(dp, row, j)) {
                dp[row] = j;
                count += intervalTotalNQueens(dp, row + 1, n);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean validNQueens(int[] dp, int row, int column) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == column || Math.abs(dp[i] - column) == Math.abs(i - row)) {
                return false;
            }
        }
        return true;
    }

    /**
     * 53. Maximum Subarray
     *
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        int result = Integer.MIN_VALUE;
        int local = Integer.MIN_VALUE;
        for (int num : nums) {
            local = local < 0 ? num : local + num;
            result = Math.max(result, local);
        }
        return result;
    }

    /**
     * 54. Spiral Matrix
     *
     * @param matrix
     * @return
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        int top = 0;
        int left = 0;
        int right = matrix[0].length - 1;
        int bottom = matrix.length - 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                result.add(matrix[top][i]);
            }
            for (int i = top + 1; i <= bottom; i++) {
                result.add(matrix[i][right]);
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    result.add(matrix[bottom][i]);
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result.add(matrix[i][left]);
                }
            }
            top++;
            bottom--;
            left++;
            right--;
        }
        return result;
    }

    /**
     * 55. Jump Game
     *
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int reach = 0;
        for (int i = 0; i < nums.length - 1 && i <= reach; i++) {
            reach = Math.max(reach, i + nums[i]);
        }
        return reach >= nums.length - 1;
    }

    /**
     * 56. Merge Intervals
     *
     * @param intervals
     * @return
     */
    public int[][] merge(int[][] intervals) {
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        LinkedList<int[]> result = new LinkedList<>();
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        for (int[] interval : intervals) {
            if (result.isEmpty() || result.peekLast()[1] < interval[0]) {
                result.add(interval);
            } else {
                int[] pre = result.peekLast();

                pre[0] = Math.min(pre[0], interval[0]);

                pre[1] = Math.max(pre[1], interval[1]);
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
        if (intervals == null || newInterval == null) {
            return new int[][]{};
        }
        Arrays.sort(intervals, Comparator.comparing(o -> o[0]));
        int endIndex = 0;
        int len = intervals.length;
        LinkedList<int[]> linkedList = new LinkedList<>();
        while (endIndex < len && intervals[endIndex][1] < newInterval[0]) {
            linkedList.offer(intervals[endIndex++]);
        }
        while (endIndex < len && intervals[endIndex][0] <= newInterval[1]) {
            newInterval[0] = Math.min(intervals[endIndex][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[endIndex][1], newInterval[1]);
            endIndex++;
        }
        linkedList.offer(newInterval);
        while (endIndex < len) {
            linkedList.offer(intervals[endIndex++]);
        }
        return linkedList.toArray(new int[][]{});
    }


    /**
     * 59. Spiral Matrix II
     *
     * @param n
     * @return
     */
    public int[][] generateMatrix(int n) {
        if (n <= 0) {
            return new int[][]{};
        }
        int[][] result = new int[n][n];
        int top = 0;
        int bottom = n - 1;
        int left = 0;
        int right = n - 1;
        int total = 1;
        while (left <= right && top <= bottom) {
            for (int i = left; i <= right; i++) {
                result[top][i] = total++;
            }
            for (int i = top + 1; i <= bottom; i++) {
                result[i][right] = total++;
            }
            if (top != bottom) {
                for (int i = right - 1; i >= left; i--) {
                    result[bottom][i] = total++;
                }
            }
            if (left != right) {
                for (int i = bottom - 1; i > top; i--) {
                    result[i][left] = total++;
                }
            }
            top++;
            right--;
            left++;
            right--;
        }
        return result;
    }

    /**
     * todo
     * 60. Permutation Sequence
     *
     * @param n
     * @param k
     * @return
     */
    public String getPermutation(int n, int k) {
        if (n <= 0) {
            return "";
        }
        List<Integer> nums = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }
        int[] position = new int[n + 1];
        position[0] = 1;
        position[1] = 1;
        int base = 1;
        for (int i = 2; i <= n; i++) {


        }
        return "";
    }


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
            current = current.next;
            count++;
        }
        current.next = head;
        k %= count;
        if (k != 0) {
            for (int i = 0; i < count - k; i++) {
                head = head.next;
                current = current.next;
            }
        }
        current.next = null;
        return head;

    }


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
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }


    public int uniquePathsV2(int m, int n) {
        if (m < 0 || n < 0) {
            return 0;
        }
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[j] = dp[j] + dp[j - 1];

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
        for (int i = 0; i < dp.length; i++) {
            if (obstacleGrid[0][i] == 1) {
                break;
            }
            dp[0] = 1;
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[j] = 0;
                } else if (j > 0) {
                    dp[j] = dp[j - 1] + dp[j];
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
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        int[] dp = new int[column];

        for (int j = 0; j < column; j++) {
            dp[j] = j == 0 ? grid[0][j] : dp[j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[j] += grid[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }
            }
        }
        return dp[column - 1];
    }


    /**
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
        boolean seenEAfterNumber = false;
        boolean seenDigit = false;
        boolean seenE = false;
        char[] words = s.toCharArray();
        for (int i = 0; i < words.length; i++) {
            char word = words[i];
            if (Character.isDigit(word)) {
                seenNumber = true;
                seenEAfterNumber = true;
            } else if (word == 'e' || word == 'E') {
                if (i == 0 || seenE || !seenNumber) {
                    return false;
                }
                if (!Character.isDigit(words[i - 1]) && words[i - 1] != '.') {
                    return false;
                }
                seenE = true;
                seenEAfterNumber = false;
            } else if (word == '-' || word == '+') {
                if (i != 0 && (words[i - 1] != 'e' && words[i - 1] != 'E')) {
                    return false;
                }
            } else if (word == '.') {
                if (i == 0) {
                    return false;
                }
                if (seenE || seenDigit) {
                    return false;
                }
                seenDigit = true;
            } else {
                return false;
            }
        }
        return seenNumber && seenEAfterNumber;
    }

    /**
     * 66. Plus One
     *
     * @param digits
     * @return
     */
    public int[] plusOne(int[] digits) {
        if (digits == null || digits.length == 0) {
            return new int[]{};
        }
        int len = digits.length - 1;
        for (int i = len; i > 0; i--) {
            if (digits[i] != 9) {
                digits[i]++;
                return digits;
            } else {
                digits[i] = 0;
            }
        }
        int[] result = new int[digits.length + 1];
        result[0] = 1;
        return result;
    }


    /**
     * 67. Add Binary
     *
     * @param a
     * @param b
     * @return
     */
    public String addBinary(String a, String b) {
        int m = a.length() - 1;
        int n = b.length() - 1;
        int carry = 0;
        StringBuilder builder = new StringBuilder();
        while (m >= 0 || n >= 0 || carry > 0) {
            int val = (m >= 0 ? Character.getNumericValue(a.charAt(m--)) : 0)
                    + (n >= 0 ? Character.getNumericValue(b.charAt(n--)) : 0) + carry;
            builder.append(val % 2);
            carry = val / 2;
        }
        return builder.reverse().toString();
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
            int line = 0;
            int endIndex = startIndex;
            while (endIndex < words.length && line + words[endIndex].length() <= maxWidth) {
                line += words[endIndex++].length() + 1;
            }
            boolean lastRow = endIndex == words.length;
            int countOfWord = endIndex - startIndex;
            int blankRow = maxWidth - line + 1;
            int blankCount = countOfWord - 1;
            StringBuilder builder = new StringBuilder();
            if (countOfWord == 1) {
                builder.append(words[startIndex]);
            } else {
                int space = lastRow ? 1 : 1 + blankRow / blankCount;
                int extra = lastRow ? 0 : blankRow % blankCount;
                builder.append(constructWord(words, space, extra, startIndex, endIndex));
            }
            String tmp = trimWord(builder.toString(), maxWidth);
            result.add(tmp);
            startIndex = endIndex;
        }
        return result;
    }

    private String constructWord(String[] words, int blankSpace, int extraSpace, int start, int end) {
        StringBuilder builder = new StringBuilder();

        for (int i = start; i < end; i++) {
            builder.append(words[i]);
            int tmp = blankSpace;
            while (tmp-- > 0) {
                builder.append(" ");
            }
            if (extraSpace-- > 0) {
                builder.append(" ");
            }
        }
        return builder.toString();
    }

    private String trimWord(String word, int maxWidth) {
        StringBuilder wordBuilder = new StringBuilder(word);

        while (wordBuilder.length() < maxWidth) {

            wordBuilder.append(" ");
        }
        word = wordBuilder.toString();
        while (word.length() > maxWidth) {
            word = word.substring(0, word.length() - 1);
        }
        return word;
    }

    /**
     * todo
     *
     * @param words
     * @param maxWidth
     * @return
     */
    public List<String> fullJustifyV2(String[] words, int maxWidth) {
        if (words == null || words.length == 0) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        int start = 0;
        while (start < words.length) {
            int line = 0;
            int index = 0;
            while (start + index < words.length && line + words[start + index].length() <= maxWidth - index) {
                line += words[start + index].length();
                index++;
            }
            StringBuilder builder = new StringBuilder();
            boolean lastRow = start + index == words.length;
            int blankWord = index - 1;
            int remain = maxWidth - line;
            for (int i = 0; i < index; i++) {
                builder.append(words[start + i]);
                if (lastRow) {
                    builder.append(" ");
                } else if (index > 1) {
                    int space = (maxWidth - line) / (index - 1) + (i < (remain % (index - 1)) ? 1 : 0);
                    while (space-- > 0) {
                        builder.append(" ");
                    }
                }
            }
            result.add(trimWord(builder.toString(), maxWidth));
            start += index;
        }
        return result;
    }

    /**
     * 69. Sqrt(x)
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
     * 70. Climbing Stairs
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n <= 0) {
            return 0;
        }
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        int step1 = 1;
        int step2 = 2;
        int result = 0;
        for (int i = 3; i <= n; i++) {
            result = step1 + step2;
            step1 = step2;
            step2 = result;
        }
        return result;
    }


    /**
     * 71. Simplify Path
     *
     * @param path
     * @return
     */
    public String simplifyPath(String path) {
        if (path == null || path.isEmpty()) {
            return "/";
        }
        String[] split = path.split("/");
        List<String> skip = Arrays.asList("", "..", "/", ".");
        LinkedList<String> deque = new LinkedList<>();
        for (String word : split) {
            if (word.equals("..")) {
                if (!deque.isEmpty()) {
                    deque.poll();
                }
            } else if (!skip.contains(word)) {
                deque.offer(word);
            }
        }
        if (deque.isEmpty()) {
            return "/";
        }
        StringBuilder result = new StringBuilder();
        for (String word : deque) {
            result.append("/").append(word);
        }
        return result.toString();
    }

    /**
     * todo
     * 72. Edit Distance
     *
     * @param word1
     * @param word2
     * @return
     */
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
                    dp[i][j] = 1 + Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]);
                }
            }
        }
        return dp[m][n];

    }


    /**
     * 73. Set Matrix Zeroes
     *
     * @param matrix
     */
    public void setZeroes(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return;
        }
        boolean setRow = false;
        boolean setColumn = false;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                boolean zeroNum = matrix[i][j] == 0;
                if (!zeroNum) {
                    continue;
                }
                matrix[i][0] = 0;
                matrix[0][j] = 0;
                if (i == 0) {
                    setColumn = true;
                }
                if (j == 0) {
                    setRow = true;
                }

            }
        }
        for (int i = 1; i < matrix.length; i++) {
            for (int j = 1; j < matrix[i].length; j++) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        if (setRow) {
            for (int i = 0; i < matrix.length; i++) {
                matrix[i][0] = 0;
            }
        }
        if (setColumn) {
            Arrays.fill(matrix[0], 0);
        }
    }


    /**
     * 74. Search a 2D Matrix
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
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


    /**
     * todo
     * 75. Sort Colors
     *
     * @param nums
     */
    public void sortColors(int[] nums) {
        if (nums == null || nums.length == 0) {
            return;
        }
        int i = 0, j = nums.length - 1;
        for (int k = 0; k <= j; ) {

        }
        return;
    }


    /**
     * 77. Combinations
     *
     * @param n
     * @param k
     * @return
     */
    public List<List<Integer>> combine(int n, int k) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        combine(result, new ArrayList<>(), 1, n, k);
        return result;
    }

    private <E> void combine(List<List<Integer>> result, List<Integer> tmp, int start, int n, int k) {
        if (tmp.size() == k) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i <= n; i++) {
            tmp.add(i);
            combine(result, tmp, i + 1, n, k);
            tmp.remove(tmp.size() - 1);
        }
    }


    /**
     * 78. Subsets
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(int[] nums) {
        if (nums == null || nums.length == 0) {
            return new ArrayList<>();
        }
        List<List<Integer>> result = new ArrayList<>();
        subsets(result, new ArrayList<Integer>(), 0, nums);
        return result;
    }

    private void subsets(List<List<Integer>> result, ArrayList<Integer> integers, int start, int[] nums) {
        result.add(new ArrayList<>(integers));
        for (int i = start; i < nums.length; i++) {
            integers.add(nums[i]);
            subsets(result, integers, i + 1, nums);
            integers.remove(integers.size() - 1);
        }
    }

    /**
     * 80. Remove Duplicates from Sorted Array II
     *
     * @param nums
     * @return
     */
    public int removeDuplicatesV2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int index = 1;
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[index++] = nums[i];
                count = 1;
            } else {
                count++;
                if (count > 2) {
                    continue;
                }
                nums[index++] = nums[i];
            }
        }
        return index;
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
     * 82. Remove Duplicates from Sorted List II
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode current = head.next;
        if (head.val == current.val) {
            while (current != null && current.val == head.val) {
                current = current.next;
            }
            return deleteDuplicates(current);
        }
        head.next = deleteDuplicates(head.next);
        return head;

    }


    /**
     * 83. Remove Duplicates from Sorted List
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicatesV2(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        if (head.val != head.next.val) {
            head.next = deleteDuplicatesV2(head.next);
            return head;
        }
        return deleteDuplicatesV2(head.next);
    }


    /**
     * 84. Largest Rectangle in Histogram
     *
     * @param heights
     * @return
     */
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
                int edgeHeight = heights[stack.pop()];

                int edge = stack.isEmpty() ? i : i - stack.peek() - 1;

                result = Math.max(result, edgeHeight * edge);

                i--;
            }
        }
        return result;
    }


    /**
     * todo
     * 85. Maximal Rectangle
     *
     * @param matrix
     * @return
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return 0;
        }
        int row = matrix.length;

        int column = matrix[0].length;

        int result = 0;

        int[] left = new int[column];

        int[] right = new int[column];

        int[] height = new int[column];

        Arrays.fill(right, column);

        for (int i = 0; i < row; i++) {
            int leftSide = 0;
            int rightSide = column;
            for (int j = 0; j < column; j++) {
                if (matrix[i][j] == '1') {
                    height[j]++;
                    left[j] = Math.max(left[j], leftSide);
                } else {
                    height[j] = 0;

                    left[j] = leftSide;

                    leftSide = j + 1;
                }
            }
            for (int j = column - 1; j >= 0; j--) {
                if (matrix[i][j] == '1') {
                    right[j] = Math.min(right[j], rightSide);
                } else {
                    right[j] = column;
                    rightSide = j;
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


    /**
     * todo
     * 87. Scramble String
     *
     * @param s1
     * @param s2
     * @return
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null || s2 == null) {
            return false;
        }
        int m = s1.length();
        int n = s2.length();
        if (m != n) {
            return false;
        }
        int[] hash = new int[256];
        for (int i = 0; i < m; i++) {
            hash[s1.charAt(i) - 'a']++;
            hash[s2.charAt(i) - 'a']--;
        }
        for (int num : hash) {
            if (num != 0) {
                return false;
            }
        }
        if (s1.equals(s2)) {
            return true;
        }
        for (int i = 1; i < m; i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(i), s2.substring(0, m - i)) && isScramble(s1.substring(0, i), s2.substring(m - i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * 88. Merge Sorted Array
     *
     * @param nums1
     * @param m
     * @param nums2
     * @param n
     */
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int k = m + n - 1;
        m--;
        n--;
        while (m >= 0 && n >= 0) {
            if (nums1[m] >= nums2[n]) {
                nums1[k--] = nums1[m--];
            } else {
                nums1[k--] = nums2[n--];
            }
        }
        while (n >= 0) {
            nums1[k--] = nums2[n--];
        }
    }


    /**
     * todo
     * 89. Gray Code
     *
     * @param n
     * @return
     */
    public List<Integer> grayCode(int n) {
        return null;
    }

    /**
     * todo
     * 91. Decode Ways
     *
     * @param s
     * @return
     */
    public int numDecodings(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        StringBuilder builder = new StringBuilder();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char tmp = s.charAt(i);
            char word = (char) ((tmp - 1) % 26 + 'A');
            builder.append(word);
        }
        int result = builder.length();
        return result == 0 ? 1 : result;
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
        if (head == null || head.next == null) {
            return head;
        }
        ListNode root = new ListNode(0);

        root.next = head;

        ListNode slow = root;

        for (int i = 0; i < m - 1; i++) {
            slow = slow.next;
        }
        ListNode start = slow.next;
        ListNode then = start.next;
        for (int i = 0; i < n - m; i++) {
            start.next = then.next;

            then.next = slow.next;

            slow.next = then;

            then = start.next;
        }
        return root.next;
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

    public ListNode reverseBetweenV2(ListNode head, int m, int n) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        for (int i = 0; i < m - 1; i++) {

        }
        return null;
    }


    /**
     * 93. Restore IP Addresses
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        int len = s.length();

        for (int i = 1; i < 4 && i < len - 2; i++) {
            for (int j = i + 1; j < i + 4 && j < len - 1; j++) {
                for (int k = j + 1; k < j + 4 && k < len; k++) {
                    String a = s.substring(0, i);
                    String b = s.substring(i, j);
                    String c = s.substring(j, k);
                    String d = s.substring(k);
                    if (validIp(a) && validIp(b) && validIp(c) && validIp(d)) {
                        String tmp = a + "." + b + "." + c + "." + d;
                        result.add(tmp);
                    }

                }
            }
        }
        return result;
    }

    private boolean validIp(String s) {
        return !s.isEmpty() && s.length() <= 3 && Integer.parseInt(s) <= 255 && (s.length() == 1 || s.charAt(0) != '0');
    }


    /**
     * todo
     *
     * @param s
     * @return
     */
    public List<String> restoreIpAddressesV2(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        for (int a = 1; a <= 3; a++) {
            for (int b = 1; b <= 3; b++) {
                for (int c = 1; c <= 3; c++) {
                    String s1 = s.substring(0, a);
                    String s2 = s.substring(a, a + b);
                    String s3 = s.substring(a + b, a + b + c);
                    String s4 = s.substring(a + b + c);
                    if (s1.length() + s2.length() + s3.length() + s4.length() == 12) {
                        if (validIp(s1) && validIp(s1) && validIp(s1) && validIp(s1)) {
                            String tmp = s1 + "." + s2 + "." + s3 + "." + s4;
                            result.add(tmp);
                        }
                    }
                }
            }
        }

        return result;
    }


    /**
     * 94. Binary Tree Inorder Traversal
     *
     * @param root
     * @return
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<>();
        }
        List<Integer> result = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        while (p != null || !stack.isEmpty()) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();
            result.add(p.val);
            p = p.right;
        }
        return result;
    }

    /**
     * 95. Unique Binary Search Trees II
     *
     * @param n
     * @return
     */
    public List<TreeNode> generateTrees(int n) {
        if (n <= 0) {
            return new ArrayList<>();
        }
        return intervalGenerateTrees(1, n);
    }

    private List<TreeNode> intervalGenerateTrees(int start, int end) {
        List<TreeNode> result = new ArrayList<>();
        if (start == end) {
            result.add(new TreeNode(start));
            return result;
        }
        if (start > end) {
            result.add(null);
            return result;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> leftNodes = intervalGenerateTrees(start, i - 1);
            List<TreeNode> rightNodes = intervalGenerateTrees(i + 1, end);

            for (TreeNode leftNode : leftNodes) {
                for (TreeNode rightNode : rightNodes) {
                    TreeNode root = new TreeNode(i);
                    root.left = leftNode;
                    root.right = rightNode;
                    result.add(root);
                }
            }
        }
        return result;
    }


    /**
     * 96. Unique Binary Search Trees
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        if (n <= 0) {
            return 0;
        }
        if (n <= 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }


    /**
     * 97. Interleaving String
     *
     * @param s1
     * @param s2
     * @param s3
     * @return
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        int m = s1.length();
        int n = s2.length();
        if (m + n != s3.length()) {
            return false;
        }
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 1; i <= m; i++) {
            dp[i][0] = s1.charAt(i - 1) == s3.charAt(i - 1) && dp[i - 1][0];
        }
        for (int j = 1; j <= n; j++) {
            dp[0][j] = s2.charAt(j - 1) == s3.charAt(j - 1) && dp[0][j - 1];
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                dp[i][j] = (s1.charAt(i - 1) == s3.charAt(i + j - 1) && dp[i - 1][j])
                        || (s2.charAt(j - 1) == s3.charAt(i + j - 1) && dp[i][j - 1]);
            }
        }
        return dp[m][n];
    }

    /**
     * 98. Validate Binary Search Tree
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            if (prev != null && prev.val >= p.val) {
                return false;
            }
            prev = p;
            p = p.right;
        }
        return true;
    }


    /**
     * 99. Recover Binary Search Tree
     *
     * @param root
     */
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        Stack<TreeNode> stack = new Stack<>();
        TreeNode p = root;
        TreeNode first = null;
        TreeNode second = null;
        TreeNode prev = null;
        while (!stack.isEmpty() || p != null) {
            while (p != null) {
                stack.push(p);
                p = p.left;
            }
            p = stack.pop();

            if (prev != null && prev.val >= p.val) {
                if (first == null) {
                    first = prev;
                }
                second = p;
            }
            prev = p;
            p = p.right;
        }
        if (first != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }
    }


}
