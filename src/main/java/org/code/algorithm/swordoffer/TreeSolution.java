package org.code.algorithm.swordoffer;

import org.code.algorithm.datastructe.TreeNode;

import java.util.Deque;
import java.util.LinkedList;

/**
 * @author luk
 * @date 2020/12/1
 */
public class TreeSolution {

    public int index = -1;

    public String Serialize(TreeNode root) {
        StringBuilder builder = new StringBuilder();
        intervalSerialize(builder, root);
        return builder.toString();
    }

    private void intervalSerialize(StringBuilder builder, TreeNode root) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val + ",");
        intervalSerialize(builder, root.left);
        intervalSerialize(builder, root.right);
    }

    public TreeNode Deserialize(String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        Deque<String> deque = new LinkedList<>();
        String[] words = str.split(",");
        for (String word : words) {
            deque.offer(word);
        }
        return intervalDeserialize(deque);
    }

    private TreeNode intervalDeserialize(Deque<String> deque) {
        if (deque.isEmpty()) {
            return null;
        }
        String poll = deque.poll();
        if (poll.equals("#")) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(poll));
        root.left = intervalDeserialize(deque);
        root.right = intervalDeserialize(deque);
        return root;
    }

//    public TreeNode DeserializeV2(String str) {
//        if (str == null || str.isEmpty()) {
//            return null;
//        }
//        String[] words = str.split(",");
//        return intervalDeserializeV2(words);
//    }
//
//    private TreeNode intervalDeserializeV2(String[] words) {
//        index++;
//        if (index == words.length) {
//            return null;
//        }
//        if (words[index].equals("#")) {
//            return null;
//        }
//        TreeNode root = new TreeNode(Integer.parseInt(words[index]));
//        root.left = intervalDeserializeV2(words);
//        root.right = intervalDeserializeV2(words);
//        return root;
//    }

}
