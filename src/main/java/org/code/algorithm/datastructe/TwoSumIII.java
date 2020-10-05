package org.code.algorithm.datastructe;

import java.util.HashMap;
import java.util.Map;

/**
 * @author luk
 * @date 2020/9/5
 */
public class TwoSumIII {

    private Map<Integer, Integer> map = new HashMap<>();

    public static void main(String[] args) {
        TwoSumIII sumIII = new TwoSumIII();
        sumIII.add(2);
        sumIII.add(3);
//        sumIII.find(4);
//        sumIII.find(5);
//        sumIII.find(6);
        sumIII.add(3);
        sumIII.find(6);
    }

    /**
     * @param number: An integer
     * @return: nothing
     */
    public void add(int number) {
        Integer count = map.getOrDefault(number, 0);
        count++;
        map.put(number, count);
        // write your code here
    }

    /**
     * @param value: An integer
     * @return: Find if there exists any pair of numbers which sum is equal to the value.
     */
    public boolean find(int value) {
        // write your code here
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            Integer number = entry.getKey();

            int remainValue = value - number;

            if (remainValue == number && entry.getValue() > 1) {
                return true;
            }
            if (remainValue != number && map.containsKey(remainValue)) {
                return true;
            }
        }
        return false;
    }
}
