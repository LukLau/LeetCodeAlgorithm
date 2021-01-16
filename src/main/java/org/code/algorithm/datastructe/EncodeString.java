package org.code.algorithm.datastructe;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author luk
 * @date 2020/11/11
 */
public class EncodeString {

    public static void main(String[] args) {
        EncodeString encodeString = new EncodeString();
        List<String> tmp = Arrays.asList("C#", "&", "~Xp|F", "R4QBf9g=_");
        String encode = encodeString.encode(tmp);
        encodeString.decode(encode);
    }

    /**
     * @param strs: a list of strings
     * @return: encodes a list of strings to a single string.
     */
    public String encode(List<String> strs) {
        // write your code here
        if (strs == null || strs.isEmpty()) {
            return "#";
        }
        StringBuilder builder = new StringBuilder();
        int size = strs.size();
        for (int i = 0; i < size; i++) {
            String word = strs.get(i);

            builder.append(word);
            if (i < size - 1) {
                builder.append(" ").append("#");
            }
        }
        return builder.toString();
    }

    /**
     * @param str: A string
     * @return: dcodes a single string to a list of strings
     */
    public List<String> decode(String str) {
        if (str == null || str.isEmpty()) {
            return new ArrayList<>();
        }
        String[] words = str.split(" #");
        int len = words.length;
        List<String> result = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            String word = words[i];
//            if (i != len - 1) {
//                word = word.substring(0, word.length() - len);
//            }
            result.add(word);
        }
        return result;
        // write your code here
    }
}
