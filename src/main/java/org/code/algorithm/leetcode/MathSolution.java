package org.code.algorithm.leetcode;

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


}
