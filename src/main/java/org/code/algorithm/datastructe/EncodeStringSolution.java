package org.code.algorithm.datastructe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author luk
 * @date 2020/11/12
 */
public class EncodeStringSolution {


    public static void main(String[] args) {
        EncodeStringSolution stringSolution = new EncodeStringSolution();

        List<String> param = Arrays.asList("lint", "code", "love", "you");

        String encode = stringSolution.encode(param);

        stringSolution.decode(encode);
    }


    /**
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    /**
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    public String encode(List<String> strs) {
        // write your code here
        if (strs == null || strs.isEmpty()) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        int size = strs.size();
        String split = " /";
        for (int i = 0; i < size; i++) {
            String content = strs.get(i);
            builder.append(content);
            if (i != size - 1) {
                builder.append(split);
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
        List<String> result = new ArrayList<>();
//        int length = str.length();
        String split = " /";
        String[] words = str.split(split);

        for (int i = 0; i < words.length; i++) {
            String word = words[i];
            // if (i != words.length - 1) {
            //     word = word.substring(0, word.length() - 1);
            // }
            result.add(word);

        }
//        Collections.addAll(result, words);
        return result;
    }

}
