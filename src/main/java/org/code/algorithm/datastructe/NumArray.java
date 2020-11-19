package org.code.algorithm.datastructe;

import com.sun.tools.javac.util.Pair;

import java.util.HashMap;
import java.util.Map;

/**
 * 303. Range Sum Query - Immutable
 *
 * @author luk
 * @date 2020/11/19
 */
public class NumArray {

    private Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();

    public NumArray(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int sum = nums[i];
            for (int j = i + 1; j < nums.length; j++) {
                sum += nums[j];
                map.put(Pair.of(i, j), sum);
            }
        }

    }

    public int sumRange(int i, int j) {
        return map.getOrDefault(Pair.of(i, j), 0);
    }

    public static void main(String[] args) {
        int[] nums = new int[]{-2, 0, 3, -5, 2, -1};
        NumArray numArray = new NumArray(nums);
        System.out.println(numArray.sumRange(0, 2));
        System.out.println(numArray.sumRange(2, 5));
        System.out.println(numArray.sumRange(0, 5));

    }
}
