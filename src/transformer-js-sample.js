// transformer_similarity.js
// 使用 transformers.js 生成词向量并计算相似度
process.env.HF_HOME = './src/models';
import { pipeline } from '@huggingface/transformers';

// 1️⃣ 加载 Transformer 句向量模型（自动下载 & 缓存）
// 这个模型是 sentence-transformers/all-MiniLM-L6-v2 的 JS/ONNX 版本
const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

// 2️⃣ 要测试的词汇（保持和 Word2Vec 一致）
const wordsToCheck = [
  'cat', 'dog', 'kitten', 'puppy',
  'apple', 'banana', 'orange', 'fruit',
  'car', 'bus', 'train', 'vehicle',
  'king', 'queen', 'man', 'woman'
];

// 3️⃣ 工具函数
function cosineSimilarity(vecA, vecB) {
  const dot = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
  const normB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
  return dot / (normA * normB);
}

function similarityLabel(sim) {
  if (sim >= 0.7) return '非常相似';
  if (sim >= 0.4) return '较为相似';
  return '不太相似';
}

// 4️⃣ 生成词向量
const vectors = {};
console.log('\n词向量示例 (前5维):');
for (const word of wordsToCheck) {
  const output = await extractor(word, { pooling: 'mean', normalize: true });
  vectors[word] = output.data;
  console.log(`${word}: [${vectors[word].slice(0, 5).map(v => v.toFixed(3)).join(', ')} ...]`);
}

// 5️⃣ 计算两两相似度
console.log('\n词相似度:');
for (let i = 0; i < wordsToCheck.length; i++) {
  for (let j = i + 1; j < wordsToCheck.length; j++) {
    const w1 = wordsToCheck[i];
    const w2 = wordsToCheck[j];
    const sim = cosineSimilarity(vectors[w1], vectors[w2]);
    console.log(`${w1} vs ${w2}: ${sim.toFixed(3)} (${similarityLabel(sim)})`);
  }
}
