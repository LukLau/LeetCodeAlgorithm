package org.code.algorithm.datastructe;

import java.util.HashMap;
import java.util.Map;

/**
 * @author luk
 * @date 2020/11/18
 */
public class ValidWordAbbr {

    private Map<String, Integer> wordMap;

    private Map<String, Integer> abbrMap;


    /**
     * @param dictionary: a list of words
     */
    public ValidWordAbbr(String[] dictionary) {
        wordMap = new HashMap<>();
        abbrMap = new HashMap<>();
        // do intialization if necessary
        for (String word : dictionary) {
            String abbrWord = constructAbbrWord(word);
            wordMap.put(word, wordMap.getOrDefault(word, 0) + 1);
            abbrMap.put(abbrWord, abbrMap.getOrDefault(word, 0) + 1);
        }
    }


    private String constructAbbrWord(String word) {

        StringBuilder builder = new StringBuilder();
        builder.append(word.charAt(0));
        int len = word.length();
        int middleLen = len - 2;
        builder.append(middleLen);
        builder.append(word.charAt(len - 1));
        return builder.toString();
    }


    /**
     * @param word: a string
     * @return: true if its abbreviation is unique or false
     */
    public boolean isUnique(String word) {
        if (word.length() <= 2) {
            return true;
        }
        String abbrWord = constructAbbrWord(word);
        return wordMap.getOrDefault(word, 0).equals(abbrMap.getOrDefault(abbrWord, 0));
    }


    public static void main(String[] args) {
        String[] words = new String[]{"deer", "door", "cake", "card"};
        ValidWordAbbr wordAbbr = new ValidWordAbbr(words);

        System.out.println(wordAbbr.isUnique("dear"));
        System.out.println(wordAbbr.isUnique("cart"));
        System.out.println(wordAbbr.isUnique("cane"));
        System.out.println(wordAbbr.isUnique("make"));

    }

}
