
// glove_similarity.js
// 使用预训练的 GloVe 向量，计算词语相似度

const fs = require("fs");
const path = require("path");

// 预训练 GloVe 文件路径（请修改为你解压后的文件路径）
const GLOVE_FILE = path.join(__dirname, "./models/glove.6B.50d.txt");

// 要测试的词汇
const wordsToCheck = [
  "cat",
  "dog",
  "kitten",
  "puppy",
  "apple",
  "banana",
  "orange",
  "fruit",
  "car",
  "bus",
  "train",
  "vehicle",
  "king",
  "queen",
  "man",
  "woman",
];

// ------------------------
// 1️⃣ 读取 GloVe 向量文件
// ------------------------
console.log("正在加载 GloVe 模型，请稍候...");

function loadGlove(filePath) {
  const lines = fs.readFileSync(filePath, "utf8").split("\n");
  const model = {};
  for (let line of lines) {
    if (!line) continue;
    const parts = line.split(" ");
    const word = parts[0];
    const vector = parts.slice(1).map(Number);
    model[word] = vector;
  }
  return model;
}

const gloveModel = loadGlove(GLOVE_FILE);
console.log("✅ 模型加载完成");

// ------------------------
// 2️⃣ 计算余弦相似度
// ------------------------
function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

// 相似度文字标签
function similarityLabel(sim) {
  if (sim >= 0.7) return "非常相似";
  if (sim >= 0.4) return "较为相似";
  return "不太相似";
}

// ------------------------
// 3️⃣ 输出词向量（前10维）
// ------------------------
console.log("\n词向量示例 (前10维):");
wordsToCheck.forEach((word) => {
  if (gloveModel[word]) {
    console.log(
      `${word}: [${gloveModel[word]
        .slice(0, 10)
        .map((v) => v.toFixed(3))
        .join(", ")} ...]`
    );
  } else {
    console.log(`${word}: ❌ 不在 GloVe 模型词表中`);
  }
});

// ------------------------
// 4️⃣ 输出两两相似度
// ------------------------
console.log("\n词相似度:");
for (let i = 0; i < wordsToCheck.length; i++) {
  for (let j = i + 1; j < wordsToCheck.length; j++) {
    const w1 = wordsToCheck[i];
    const w2 = wordsToCheck[j];
    if (gloveModel[w1] && gloveModel[w2]) {
      const sim = cosineSimilarity(gloveModel[w1], gloveModel[w2]);
      console.log(
        `${w1} vs ${w2}: ${sim.toFixed(3)} (${similarityLabel(sim)})`
      );
    }
  }
}
