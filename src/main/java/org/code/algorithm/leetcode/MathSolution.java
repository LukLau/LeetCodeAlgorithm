package org.code.algorithm.leetcode;

import java.util.HashMap;
import java.util.Map;

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


}
