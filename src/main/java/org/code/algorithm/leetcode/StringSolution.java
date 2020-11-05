package org.code.algorithm.leetcode;

import java.util.*;

/**
 * @author luk
 * @date 2020/10/27
 */
public class StringSolution {

    public static void main(String[] args) {
        StringSolution solution = new StringSolution();
        String word = "aabb";
        solution.generatePalindromes(word);
    }

    // ---kmp问题--- //

    /**
     * todo
     * 214. Shortest Palindrome
     *
     * @param s
     * @return
     */
    public String shortestPalindrome(String s) {
        return "";
    }


    /**
     * 300. Longest Increasing Subsequence
     *
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int len = nums.length;
        int[] dp = new int[len];
        Arrays.fill(dp, 1);
        for (int i = 1; i < len; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i] && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                }
            }
        }
        int result = 0;
        for (int num : dp) {
            result = Math.max(result, num);
        }
        return result;
    }


    // ---回文系列---//


    /**
     * @param s: the given string
     * @return: all the palindromic permutations (without duplicates) of it
     */
    public List<String> generatePalindromes(String s) {
        List<String> result = new ArrayList<>();
        if (s == null || s.isEmpty()) {
            return result;
        }
        Map<Character, Integer> map = new HashMap<>();
        char[] words = s.toCharArray();
        for (char word : words) {
            Integer count = map.getOrDefault(word, 0);

            count++;

            map.put(word, count);
        }
        StringBuilder midString = new StringBuilder();
        StringBuilder edgeString = new StringBuilder();
        for (Map.Entry<Character, Integer> characterIntegerEntry : map.entrySet()) {
            Integer value = characterIntegerEntry.getValue();
            char word = characterIntegerEntry.getKey();
            if (value % 2 == 1) {
                midString.append(word);
            }
            int count = value / 2;
            for (int i = 0; i < count; i++) {
                edgeString.append(word);
            }
            if (midString.length() > 1) {
                return result;
            }
        }
        generateCombine(edgeString.toString().toCharArray(), 0, midString.toString(), result);
        return result;
    }

    private void generateCombine(char[] words, int start, String s, List<String> result) {
        if (start == words.length - 1) {
            String word = String.valueOf(words) + s + new StringBuilder(String.valueOf(words)).reverse().toString();
            result.add(word);
            return;
        }
        for (int i = start; i < words.length; i++) {
            if (i > start && words[i] == words[start]) {
                continue;
            }
            swap(words, i, start);
            generateCombine(words, start + 1, s, result);
            swap(words, i, start);
        }
    }

    private void swap(char[] words, int i, int j) {
        char tmp = words[i];
        words[i] = words[j];
        words[j] = tmp;
    }

}
