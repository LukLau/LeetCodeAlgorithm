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

        for (int number : nums) {
            if (number > lower) {
                String range = findRange(lower, number > lower + 1 ? number - 1 : lower);
                result.add(range);
            }
            if (number == upper) {
                return result;
            }
            lower = number + 1;
        }
        if (lower <= upper) {
            result.add(findRange(lower, upper));
        }
        return result;
        // write your code here
        // write your code here
    }

    private String findRange(long lower, long upper) {
        return lower == upper ? upper + "" : lower + "->" + upper;
    }

    public int compareVersion(String version1, String version2) {

        String[] split1 = version1.split("\\.");
        String[] split2 = version2.split("\\.");

        int index1 = 0;
        int index2 = 0;
        while (index1 < split1.length || index2 < split2.length) {

            String word1 = index1 < split1.length ? split1[index1] : "0";

            String word2 = index2 < split2.length ? split2[index2] : "0";


            Integer value1 = Integer.parseInt(word1);
            Integer value2 = Integer.parseInt(word2);

            int compareTo = value1.compareTo(value2);

            if (compareTo != 0) {
                return compareTo;
            }
            index1++;
            index2++;
        }
        return 0;
    }

    /**
     * todo
     * 166. Fraction to Recurring Decimal
     *
     * @param numerator
     * @param denominator
     * @return
     */
    public String fractionToDecimal(int numerator, int denominator) {
        return "";
    }


    public int[] twoSum(int[] numbers, int target) {
        int[] result = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                Integer index1 = map.get(target - numbers[i]);
                result[0] = index1 + 1;

                result[1] = i + 1;
                return result;
            }
            map.put(numbers[i], i);
        }
        return result;
    }


    public String convertToTitle(int n) {
        StringBuilder builder = new StringBuilder();
        while (n != 0) {

            int value = (n - 1) % 26;

            char word = (char) (value + 'A');

            builder.append(word);

            n = (n - 1) / 26;

        }
        return builder.reverse().toString();
    }


    /**
     * 169 摩尔投票法
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 1;
        for (int i = 1; i < nums.length; i++) {
            int value = nums[i];
            if (value == candidate) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    count = 1;
                    candidate = value;
                }
            }
        }
        return candidate;

    }


}
