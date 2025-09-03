// word_vectors_label.js
const tf = require("@tensorflow/tfjs");

// ------------------------
// 1️⃣ 准备小语料
// ------------------------
const corpus = [
  ["cat", "dog", "cat", "kitten", "dog", "puppy"],
  ["apple", "banana", "orange", "fruit", "apple", "banana"],
  ["cat", "dog", "pet", "animal", "dog", "cat"],
];

// ------------------------
// 2️⃣ 构建词表
// ------------------------
const allWords = [...new Set(corpus.flat())];
const word2index = Object.fromEntries(allWords.map((w, i) => [w, i]));
const index2word = Object.fromEntries(allWords.map((w, i) => [i, w]));

const vocabSize = allWords.length;
const embeddingDim = 5; // 向量维度

// ------------------------
// 3️⃣ 初始化 embedding 矩阵
// ------------------------
let embeddings = tf.variable(tf.randomNormal([vocabSize, embeddingDim]));

// ------------------------
// 4️⃣ 查询词向量函数
// ------------------------
function getVector(word) {
  const idx = word2index[word];
  return embeddings.gather([idx]);
}

// ------------------------
// 5️⃣ 计算余弦相似度函数
// ------------------------
function cosineSimilarity(vecA, vecB) {
  const dot = tf.sum(tf.mul(vecA, vecB));
  const normA = tf.norm(vecA);
  const normB = tf.norm(vecB);
  return dot.div(normA.mul(normB)).dataSync()[0];
}

// ------------------------
// 6️⃣ 相似度文字标签
// ------------------------
function similarityLabel(sim) {
  if (sim >= 0.7) return "非常相似";
  if (sim >= 0.4) return "较为相似";
  return "不太相似";
}

// ------------------------
// 7️⃣ 输出词向量
// ------------------------
const wordsToCheck = ["cat", "dog", "apple", "banana", "fruit"];

console.log("\n词向量:");
wordsToCheck.forEach((word) => {
  const vec = getVector(word).dataSync();
  console.log(
    `${word}: [${Array.from(vec)
      .map((v) => v.toFixed(3))
      .join(", ")}]`
  );
});

// ------------------------
// 8️⃣ 输出词相似度 + 标签
// ------------------------
console.log("\n词相似度:");
for (let i = 0; i < wordsToCheck.length; i++) {
  for (let j = i + 1; j < wordsToCheck.length; j++) {
    const w1 = wordsToCheck[i];
    const w2 = wordsToCheck[j];
    const sim = cosineSimilarity(getVector(w1), getVector(w2));
    console.log(`${w1} vs ${w2}: ${sim.toFixed(3)} (${similarityLabel(sim)})`);
  }
}
