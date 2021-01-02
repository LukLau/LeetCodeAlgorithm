package org.code.algorithm.datastructe;

/**
 * 前缀查找树
 *
 * @author luk
 * @date 2020/10/22
 */
public class Trie {


    private final TrieNode root;

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        this.root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        if (word == null) {
            return;
        }
        TrieNode p = root;
        char[] words = word.toCharArray();
        for (char tmp : words) {
            int index = tmp - 'a';
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
        int m = word.length();
        for (int i = 0; i < m; i++) {
            char tmp = word.charAt(i);
            if (p.words[tmp - 'a'] == null) {
                return false;
            }
            p = p.words[tmp - 'a'];
        }
        return word.equals(p.word);
    }

    public boolean searchV2(String word) {
        if (word == null || word.isEmpty()) {
            return false;
        }
        return intervalSearchV2(root, word);
    }

    private boolean intervalSearchV2(TrieNode root, String word) {
        int m = word.length();

        TrieNode q = root;

        for (int i = 0; i < m; i++) {
            if (word.charAt(i) == '.') {
                for (TrieNode trieNode : q.words) {
                    if (trieNode != null && intervalSearchV2(trieNode, word.substring(i + 1))) {
                        return true;
                    }
                }
                return false;
            } else {
                int index = word.charAt(i) - 'a';

                if (q.words[index] == null) {
                    return false;
                }
                q = q.words[index];
            }
        }
        return !"".equals(q.word);

    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        if (prefix == null || prefix.isEmpty()) {
            return false;
        }
        char[] prefixWords = prefix.toCharArray();
        TrieNode p = root;
        int m = prefix.length();
        for (int i = 0; i < m; i++) {
            char tmp = prefix.charAt(i);
            if (p.words[tmp - 'a'] == null) {
                return false;
            }
            p = p.words[tmp - 'a'];
        }
        return true;
    }

    static class TrieNode {
        private TrieNode[] words = new TrieNode[26];
        private String word = "";
    }
}
