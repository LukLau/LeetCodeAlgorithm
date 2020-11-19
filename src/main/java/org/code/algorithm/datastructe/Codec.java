package org.code.algorithm.datastructe;

/**
 * 297. Serialize and Deserialize Binary Tree
 *
 * @author luk
 * @date 2020/11/19
 */
public class Codec {


    private int index = -1;

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) {
            return "#,";
        }
        return intervalSerialize(new StringBuilder(), root);
    }

    private String intervalSerialize(StringBuilder builder, TreeNode root) {
        if (root == null) {
            builder.append("#");
            return builder.toString();
        }
        builder.append(root.val).append(",");
        intervalSerialize(builder, root.left);
        intervalSerialize(builder, root.right);
        return builder.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data == null || data.isEmpty()) {
            return null;
        }
        String[] words = data.split(",");
        return intervalDeserialize(words);
    }

    private TreeNode intervalDeserialize(String[] words) {
        index++;
        if (index == words.length) {
            return null;
        }
        if (words[index].equals("#")) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(words[index]));
        root.left = intervalDeserialize(words);
        root.right = intervalDeserialize(words);
        return root;
    }
}
