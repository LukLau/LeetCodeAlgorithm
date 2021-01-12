package org.code.algorithm.datastructe;

import java.util.HashMap;
import java.util.Map;

/**
 * @author luk
 * @date 2020/11/18
 */
public class ValidWordAbbr {

    private final Map<String, Integer> map = new HashMap<>();

    private final Map<String, Integer> abbrMap = new HashMap<>();

    /**
     * @param dictionary: a list of words
     */
    public ValidWordAbbr(String[] dictionary) {
        // do intialization if necessary
        for (String word : dictionary) {
            Integer count = map.getOrDefault(word, 0);
            map.put(word, count + 1);
            String abbrWord = getAbbrWord(word);
            Integer num = abbrMap.getOrDefault(abbrWord, 0);
            abbrMap.put(abbrWord, num + 1);
        }
    }

    private String getAbbrWord(String word) {
        int len = word.length();
        if (len <= 2) {
            return word;
        }
        return String.valueOf(word.charAt(0)) +
                (len - 2) +
                word.charAt(len - 1);
    }

    /**
     * @param word: a string
     * @return: true if its abbreviation is unique or false
     */
    public boolean isUnique(String word) {
        String abbrWord = getAbbrWord(word);
        return map.getOrDefault(word, 0).equals(abbrMap.getOrDefault(abbrWord, 0));
        // write your code here
    }

    public static void main(String[] args) {
        String[] words = new String[]{"a"};
        ValidWordAbbr validWordAbbr = new ValidWordAbbr(words);

        validWordAbbr.isUnique("a");
    }


}
