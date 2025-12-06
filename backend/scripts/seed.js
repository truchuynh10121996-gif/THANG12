require('dotenv').config();
const mongoose = require('mongoose');
const QA = require('../models/qa.model');
const seedData = require('../data/seed-qa.json');

async function seedDatabase() {
  try {
    // Kết nối database
    const mongoURI = process.env.MONGODB_URI || 'mongodb://localhost:27017/agribank-digital-guard';
    await mongoose.connect(mongoURI);

    console.log('✅ Connected to MongoDB');

    // Xóa dữ liệu cũ (optional - comment out nếu không muốn xóa)
    // await QA.deleteMany({});
    // console.log('🗑️  Cleared existing Q&A data');

    // Insert seed data
    const result = await QA.insertMany(seedData);

    console.log(`✅ Successfully seeded ${result.length} Q&A scenarios`);
    console.log('\nCategories:');

    const categories = await QA.aggregate([
      { $group: { _id: '$category', count: { $sum: 1 } } }
    ]);

    categories.forEach(cat => {
      console.log(`   - ${cat._id}: ${cat.count} scenarios`);
    });

    console.log('\nLanguages:');
    const languages = await QA.aggregate([
      { $group: { _id: '$language', count: { $sum: 1 } } }
    ]);

    languages.forEach(lang => {
      console.log(`   - ${lang._id}: ${lang.count} scenarios`);
    });

    // Đóng kết nối
    await mongoose.disconnect();
    console.log('\n✅ Database seeding completed successfully!');

    process.exit(0);

  } catch (error) {
    console.error('❌ Error seeding database:', error);
    process.exit(1);
  }
}

// Run seed
seedDatabase();
