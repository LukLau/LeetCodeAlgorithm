package org.code.algorithm.leetcode;

import java.util.*;

/**
 * 数学解决方案
 *
 * @author luk
 * @date 2020/10/15
 */
public class MathSolution {

    /**
     * 136. Single Number
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int result = 0;
        for (int num : nums) {

            result ^= num;
        }
        return result;
    }

    /**
     * 137. Single Number II
     *
     * @param nums
     * @return
     */
    public int singleNumberV2(int[] nums) {
        return -1;
    }


    /**
     * 149. Max Points on a Line
     *
     * @param points
     * @return
     */
    public int maxPoints(int[][] points) {
        if (points == null || points.length == 0) {
            return 0;
        }

        int result = 0;
        for (int i = 0; i < points.length; i++) {
            HashMap<Integer, Map<Integer, Integer>> map = new HashMap<>();
            int repeatPoint = 0;
            int distinctPoint = 0;
            for (int j = i + 1; j < points.length; j++) {
                int x = points[j][0] - points[i][0];
                int y = points[j][1] - points[i][1];

                if (x == 0 && y == 0) {
                    repeatPoint++;
                    continue;
                }
                int gcd = gcd(x, y);
                x /= gcd;
                y /= gcd;
                if (map.containsKey(x)) {
                    Map<Integer, Integer> vertical = map.get(x);

                    if (vertical.containsKey(y)) {
                        Integer point = vertical.get(y);
                        vertical.put(y, point + 1);
                    } else {
                        vertical.put(y, 1);
                    }
                } else {
                    Map<Integer, Integer> vertical = new HashMap<>();
                    vertical.put(y, 1);
                    map.put(x, vertical);
                }
                distinctPoint = Math.max(distinctPoint, map.get(x).get(y));
            }
            result = Math.max(result, distinctPoint + repeatPoint + 1);
        }
        return result;
    }


    private int gcd(int x, int y) {
        if (y == 0) {
            return x;
        }
        return gcd(y, x % y);
    }

    public static void main(String[] args) {
        MathSolution solution = new MathSolution();
        int[] nums = new int[]{3, 2, 3};

        solution.majorityElement(nums);

    }


    /**
     * 152. Maximum Product Subarray
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        int maxValue = nums[0];
        int minValue = nums[0];
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int currentValue = nums[i];
            int tmpMaxValue = Math.max(Math.max(maxValue * currentValue, minValue * currentValue), currentValue);
            int tmpMinValue = Math.min(Math.min(maxValue * currentValue, minValue * currentValue), currentValue);

            result = Math.max(result, tmpMaxValue);
            maxValue = tmpMaxValue;
            minValue = tmpMinValue;
        }
        return result;
    }


    /**
     * 161 One Edit Distance
     *
     * @param s: a string
     * @param t: a string
     * @return: true if they are both one edit distance apart or false
     */
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
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                count++;
            }
            if (count > 1) {
                return false;
            }
        }
        return true;
    }


    /**
     * 163 Missing Ranges
     *
     * @param nums:  a sorted integer array
     * @param lower: An integer
     * @param upper: An integer
     * @return: a list of its missing ranges
     */
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        // write your code here
        if (nums == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();

        for (int num : nums) {
            if (num > lower) {
                String range = constructRange(lower, num - 1);

                result.add(range);
            }
            if (num == upper) {
                return result;
            }
            lower = num + 1;
        }
        if (lower <= upper) {
            String word = constructRange(lower, upper);
            result.add(word);
        }
        return result;
    }

    private String constructRange(int start, int end) {
        return start == end ? start + "" : start + "->" + end;
    }


    /**
     * 摩尔投票法
     * 169. Majority Element
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int candidate = nums[0];
        int count = 0;
        for (int num : nums) {
            if (num == candidate) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                candidate = num;
                count = 1;
            }
        }
        return candidate;
    }


    /**
     * 根据5的个数有关
     * 172. Factorial Trailing Zeroes
     *
     * @param n
     * @return
     */
    public int trailingZeroes(int n) {
        if (n <= 0) {
            return 0;
        }
        int count = 0;
        while ((n / 5) != 0) {
            count += (n / 5);
            n /= 5;
        }
        return count;
    }


    public String largestNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return "";
        }
        String[] words = new String[nums.length];
        for (int i = 0; i < words.length; i++) {
            words[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(words, (o1, o2) -> {
            String word1 = o1 + o2;
            String word2 = o2 + o1;
            return word2.compareTo(word1);
        });
        if ("0".equals(words[0])) {
            return "0";
        }
        StringBuilder builder = new StringBuilder();
        for (String word : words) {
            builder.append(word);
        }
        return builder.toString();
    }


    /**
     * todo
     * 187. Repeated DNA Sequences
     *
     * @param s
     * @return
     */
    public List<String> findRepeatedDnaSequences(String s) {
        return null;
    }


}
