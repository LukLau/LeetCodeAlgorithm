package org.code.algorithm.datastructe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author luk
 * @date 2020/11/11
 */
public class EncodeString {


    /**
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    public String encode(List<String> strs) {
        // write your code here
        if (strs == null || strs.isEmpty()) {
            return "";
        }
        String encodeFormat = "#";
        StringBuilder builder = new StringBuilder();
        int size = strs.size();
        for (int i = 0; i < size; i++) {
            String content = strs.get(i);

            builder.append(content);

            if (i != size - 1) {
                builder.append(" ").append(encodeFormat);
            }
        }
        return builder.toString();
    }

    /**
     * @param str: A string
     * @return: dcodes a single string to a list of strings
     */
    public List<String> decode(String str) {
        // write your code here
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        String[] words = str.split(" #");

        List<String> result = new ArrayList<>();

        result.addAll(Arrays.asList(words));
        return result;
    }

    public static void main(String[] args) {
        EncodeString encodeString = new EncodeString();

    }

}
