package org.code.algorithm.datastructe;

/**
 * 前缀查找树
 *
 * @author luk
 * @date 2020/10/22
 */
public class Trie {

    TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        if (word == null || word.isEmpty()) {
            return;
        }
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {
            int index = word.charAt(i) - 'a';
            if (p.words[index] == null) {
                p.words[index] = new TrieNode();
            }
            p = p.words[index];
        }
        p.word = word;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        if (word == null || word.isEmpty()) {
            return true;
        }
        TrieNode p = root;
        for (int i = 0; i < word.length(); i++) {

            int index = word.charAt(i) - 'a';
            if (p.words[index] == null) {
                return false;
            }
            p = p.words[index];

        }
        return word.equals(p.word);
    }

    public boolean searchV2(String word) {
        if (word == null || word.isEmpty()) {
            return true;
        }
        return intervalSearchV2(root, word);
    }

    private boolean intervalSearchV2(TrieNode root, String word) {
        for (int i = 0; i < word.length(); i++) {
            char tmp = word.charAt(i);
            if (tmp == '.') {
                for (TrieNode trieNode : root.words) {
                    if (trieNode != null) {
                        if (intervalSearchV2(trieNode, word.substring(i + 1))) {
                            return true;
                        }
                    }
                }
                return false;
            } else {
                int index = word.charAt(i) - 'a';
                if (root.words[index] == null) {
                    return false;
                }
                root = root.words[index];
            }
        }
        return !"".equals(root.word);
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        if (prefix == null || prefix.length() == 0) {
            return true;
        }
        TrieNode p = root;
        for (int i = 0; i < prefix.length(); i++) {
            int index = prefix.charAt(i) - 'a';
            if (p.words[index] == null) {
                return false;
            }
            p = p.words[index];
        }
        return true;
    }

    static class TrieNode {
        private TrieNode[] words = new TrieNode[26];
        private String word = "";
    }
}
