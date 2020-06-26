package org.code.algorithm.page;

import org.code.algorithm.datastructe.ListNode;

import java.util.*;
import java.util.concurrent.CountDownLatch;

/**
 * @author liulu12@xiaomi.com
 * @date 2020/6/25
 */
public class FirstPage {

    private int beginPoint = 0;
    private int longestLen = Integer.MIN_VALUE;

    public static void main(String[] args) {
        FirstPage page = new FirstPage();
        ListNode node = new ListNode(0);
        page.removeNthFromEnd(node, 1);
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
        ListNode reverseKGroup = reverseKGroup(node, k);

        return reverseList(head,reverseKGroup);

    }

    private ListNode reverseList(ListNode start, ListNode reverseKGroup) {

    }


}
