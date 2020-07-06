package org.code.algorithm.page;

import org.code.algorithm.datastructe.ListNode;

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
        int[][] result = new int[][]{{1, 2},
                {1, 1}};

        page.minPathSum(result);
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
     * @param str
     * @return
     */
    public int myAtoi(String str) {
        if (str == null) {
            return 0;
        }
        str = str.trim();

        if (str.isEmpty()) {
            return 0;
        }
        int sign = 1;

        int index = 0;

        char firstCharacter = str.charAt(index);

        if (firstCharacter == '-' || firstCharacter == '+') {
            sign = firstCharacter == '-' ? -1 : 1;
            index++;
        }
        long result = 0L;
        while (index < str.length() && Character.isDigit(str.charAt(index))) {
            result = result * 10 + Character.getNumericValue(str.charAt(index));

            if (result > Integer.MAX_VALUE) {
                return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }

            index++;
        }
        return (int) (sign * result);
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
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        int target = 0;
        int length = nums.length;
        for (int i = 0; i < length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = length - 1;
            while (left < right) {

                int val = nums[i] + nums[left] + nums[right];

                if (val == target) {
                    List<Integer> tmp = Arrays.asList(nums[i], nums[left], nums[right]);
                    while (left < right && nums[left] == nums[left + 1]) {
                        left++;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        right--;
                    }
                    left++;
                    right--;

                    result.add(tmp);
                } else if (val < target) {
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
        String[] map = new String[]{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

        LinkedList<String> result = new LinkedList<>();

        result.add("");

        int length = digits.length();

        for (int i = 0; i < length; i++) {

            int index = Character.getNumericValue(digits.charAt(i));

            String word = map[index];

            while (result.peek().length() == i) {

                String poll = result.poll();

                for (char tmp : word.toCharArray()) {
                    result.offer(poll + tmp);
                }
            }
        }
        return result;
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
        boolean positiveSign = (dividend > 0 && divisor > 0)
                || (dividend < 0 && divisor < 0);

        int sign = 1;

        if (!positiveSign) {
            sign = -1;
        }
        long result = 0;
        long dvd = Math.abs((long) dividend);
        long dvs = Math.abs((long) divisor);

        while (dvd >= dvs) {
            int count = 1;
            long tmp = dvs;
            while (dvd >= (tmp << 1)) {
                tmp <<= 1;
                count <<= 1;
            }
            dvd -= tmp;
            result += count;
        }
        return (int) (sign * result);
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
        Stack<Integer> stack = new Stack<>();
        int result = 0;
        int len = s.length();
        int left = -1;
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (word == '(') {
                stack.push(i);
            } else {
                if (!stack.isEmpty() && s.charAt(stack.peek()) == '(') {
                    stack.pop();
                } else {
                    left = i;
                }
                if (stack.isEmpty()) {
                    result = Math.max(result, i - left);
                } else {
                    result = Math.max(result, i - stack.peek());
                }
            }
        }
        return result;
    }

    public int longestValidParenthesesV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char word = s.charAt(i);
            if (stack.isEmpty() || word == '(') {
                stack.push(i);
            } else if (s.charAt(stack.peek()) == '(') {
                stack.pop();
            } else {
                stack.push(i);
            }
        }
        if (stack.isEmpty()) {
            return s.length();
        } else {
            int result = 0;
            int edge = s.length();
            while (!stack.isEmpty()) {
                Integer pop = stack.pop();

                result = Math.max(result, edge - 1 - pop);

                edge = pop;
            }
            result = Math.max(result, edge);
            return result;
        }
    }


    /**
     * 33. 搜索旋转排序数组
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] < nums[mid]) {
                if (nums[left] <= target && target < nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return -1;
    }


    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置
     *
     * @param nums
     * @param target
     * @return
     */
    public int[] searchRange(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return new int[]{-1, -1};
        }
        int[] result = new int[]{-1, -1};
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
            return result;
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
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[left] < target) {
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
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        intervalCombination(result, new ArrayList<Integer>(), 0, candidates, target);
        return result;
    }

    private void intervalCombination(List<List<Integer>> result,
                                     ArrayList<Integer> integers, int start, int[] candidates, int target) {
        if (target == 0) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = start; i < candidates.length && candidates[i] <= target; i++) {
            if (i > start && candidates[i] == candidates[i - 1]) {
                continue;
            }
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
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        intervalCombinationSums(result, new ArrayList<Integer>(), 0, candidates, target);
        return result;
    }

    private void intervalCombinationSums(List<List<Integer>> result, ArrayList<Integer> tmp,
                                         int start, int[] candidates, int target) {
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
        if (nums == null || nums.length == 0) {
            return -1;
        }
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] >= 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
                swap(nums, i, nums[i] - 1);
            }
        }
        for (int i = 0; i < nums.length; i++) {
            if (i + 1 != nums[i]) {
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
        int leftEdge = 0;
        int rightEdge = 0;
        while (left < right) {
            if (height[left] <= height[right]) {
                if (leftEdge <= height[left]) {
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

    /**
     * @param height
     * @return
     */
    public int trapV2(int[] height) {
        if (height == null || height.length == 0) {
            return 0;
        }

        int result = 0;
        int left = 0;

        int right = height.length - 1;

        while (left < right) {
            while (left < right && height[left] == 0) {
                left++;
            }
            while (left < right && height[right] == 0) {
                right--;
            }
            int edge = Math.min(height[left], height[right]);

            for (int i = left; i <= right; i++) {
                int value = height[i];

                if (value > edge) {
                    height[i] -= edge;
                } else {
                    result += edge - height[i];
                    height[i] = 0;
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
        int len1 = num1.length();

        int len2 = num2.length();

        int[] nums = new int[num1.length() + num2.length()];

        for (int i = len1 - 1; i >= 0; i--) {
            for (int j = len2 - 1; j >= 0; j--) {
                int value = Character.getNumericValue(num1.charAt(i))
                        * Character.getNumericValue(num2.charAt(j)) + nums[i + j + 1];

                nums[i + j + 1] = value % 10;

                nums[i + j] += value / 10;
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
            furthest = Math.max(furthest, i + nums[i]);

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
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        intervalPermute(result, new ArrayList<Integer>(), nums, used);
        return result;
    }

    private void intervalPermute(List<List<Integer>> result, ArrayList<Integer> tmp, int[] nums, boolean[] used) {
        if (tmp.size() == nums.length) {
            result.add(new ArrayList<>(tmp));
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
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        boolean[] used = new boolean[nums.length];
        intervalPermuteUnique(result, new ArrayList<Integer>(), used, nums);
        return result;

    }

    private void intervalPermuteUnique(List<List<Integer>> result, ArrayList<Integer> integers, boolean[] used, int[] nums) {
        if (integers.size() == nums.length) {
            result.add(new ArrayList<>(integers));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && !used[i]) {
                continue;
            }
            used[i] = true;
            integers.add(nums[i]);
            intervalPermuteUnique(result, integers, used, nums);
            integers.remove(integers.size() - 1);
            used[i] = false;
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
        char[][] matrix = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = '.';
            }
        }
        List<List<String>> result = new ArrayList<>();
        intervalSolveNQueens(result, 0, matrix);
        return result;
    }

    private void intervalSolveNQueens(List<List<String>> result, int row, char[][] matrix) {
        if (row == matrix.length) {
            List<String> tmp = new ArrayList<>();
            for (char[] chars : matrix) {
                tmp.add(String.valueOf(chars));
            }
            result.add(tmp);
            return;
        }
        for (int i = 0; i < matrix.length; i++) {
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
        for (int i = row - 1, j = column + 1; i >= 0 && j < matrix.length; i--, j++) {
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
        for (int i = dp.length - 1; i >= 0; i--) {
            dp[i] = -1;
        }
        return intervalTotalNQueens(dp, 0, n);
    }

    private int intervalTotalNQueens(int[] dp, int row, int n) {
        if (row == n) {
            return 1;
        }
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (validNQueens(dp, i, row)) {
                dp[row] = i;
                count += intervalTotalNQueens(dp, row + 1, i);
                dp[row] = -1;
            }
        }
        return count;
    }

    private boolean validNQueens(int[] dp, int row, int column) {
        for (int i = row - 1; i >= 0; i--) {
            if (dp[i] == column || Math.abs(i - row) == Math.abs(dp[i] - column)) {
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
        for (int i = 0; i < nums.length && i <= reach; i++) {
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
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));

        List<int[]> result = new ArrayList<>();

        for (int[] interval : intervals) {
            if (!result.isEmpty() && result.get(result.size() - 1)[1] >= interval[0]) {
                result.get(result.size() - 1)[1] = Math.max(result.get(result.size() - 1)[1], interval[1]);
            } else {
                result.add(interval);
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
        if (intervals == null || intervals.length == 0) {
            return new int[][]{};
        }
        Arrays.sort(intervals, Comparator.comparingInt(o -> o[0]));
        LinkedList<int[]> result = new LinkedList<>();

        int index = 0;
        int len = intervals.length;
        while (index < len && intervals[index][1] < newInterval[0]) {
            result.offer(intervals[index]);
            index++;
        }
        while (index < len && intervals[index][0] <= newInterval[1]) {
            newInterval[0] = Math.min(intervals[index][0], newInterval[0]);
            newInterval[1] = Math.max(intervals[index][1], newInterval[1]);
            index++;
        }
        result.offer(newInterval);
        while (index < len) {
            result.offer(intervals[index++]);
        }
        return result.toArray(new int[][]{});
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
        if (head == null || k == 0) {
            return head;
        }
        ListNode fast = head;
        int count = 1;
        while (fast.next != null) {
            fast = fast.next;
            count = count + 1;
        }
        if ((k %= count) != 0) {
            fast.next = head;
            for (int i = 0; i < count - k; i++) {
                head = head.next;
                fast = fast.next;
            }
            fast.next = null;
        }
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

        dp[0] = grid[0][0];

        for (int j = 1; j < column; j++) {
            dp[j] = dp[j - 1] + grid[0][j];
        }
        for (int i = 1; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (j == 0) {
                    dp[j] = dp[j] + grid[i][j];
                } else {
                    dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
                }
            }
        }
        return dp[column - 1];
    }


}
