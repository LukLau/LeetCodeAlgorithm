package org.code.algorithm.leetcode.daily;

import org.code.algorithm.datastructe.Node;

import java.util.ArrayList;
import java.util.List;

/**
 * @author dora
 * @date 2020/8/10
 */
public class Solution {


    /**
     * 696. 计数二进制子串
     *
     * @param s
     * @return
     * @date 08/10
     */
    public int countBinarySubstrings(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        List<Integer> counts = new ArrayList<>();
        int index = 0;
        while (index < s.length()) {
            char word = s.charAt(index);
            int count = 0;
            while (index < s.length() && s.charAt(index) == word) {
                index++;
                count++;
            }
            counts.add(count);
        }
        int result = 0;
        for (int i = 1; i < counts.size(); i++) {
            result += Math.min(counts.get(i - 1), counts.get(i));
        }
        return result;
    }

    public int countBinarySubstringsV2(String s) {
        if (s == null || s.isEmpty()) {
            return 0;
        }
        int size = s.length();
        int last = 0;
        int result = 0;

        int index = 0;
        while (index < size) {
            int count = 0;
            char word = s.charAt(index);
            while (index < size && s.charAt(index) == word) {
                index++;
                count++;
            }
            result += Math.min(last, count);
            last = count;
        }
        return result;
    }


    /**
     * todo
     * 133. 克隆图
     *
     * @param node
     * @return
     */
    public Node cloneGraph(Node node) {
        return null;
    }


}
