package org.code.algorithm.datastructe;

/**
 * @author luk
 * @date 2020/10/22
 */
public class WordDictionary {
    private Trie root;

    /**
     * Initialize your data structure here.
     */
    public WordDictionary() {
        root = new Trie();
    }

    public static void main(String[] args) {
        WordDictionary wordDictionary = new WordDictionary();
        wordDictionary.addWord("at");
        wordDictionary.search(".at");
    }

    /**
     * Adds a word into the data structure.
     */
    public void addWord(String word) {
        root.insert(word);
    }

    /**
     * Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
     */
    public boolean search(String word) {
        return root.searchV2(word);
    }

}
