#include "gtest/gtest.h"
#include "./tensor/tensor.h"
#include "./tensor/tensor.cpp"

TEST(ShapeTest, ConstructorAndGetRank) {

  Shape s({2, 3, 4});

  ASSERT_EQ(s.get_rank(), 3);
  ASSERT_EQ(s.total_elements, 24);
}

TEST(TensorTestDouble, ConstructorAndAccess) {

  Tensor<double> t({2, 3}, Layout::kNd);
  t.at({0, 0}) = 1.0;
  t.at({0, 1}) = 2.0;
  t.at({0, 2}) = 3.0;
  t.at({1, 0}) = 4.0;
  t.at({1, 1}) = 5.0;
  t.at({1, 2}) = 6.0;

  ASSERT_DOUBLE_EQ(t.at({0, 0}), 1.0);  
  ASSERT_DOUBLE_EQ(t.at({0, 1}), 2.0);
  ASSERT_DOUBLE_EQ(t.at({0, 2}), 4.0);
  ASSERT_DOUBLE_EQ(t.at({1, 0}), 4.0);
  ASSERT_DOUBLE_EQ(t.at({1, 1}), 5.0);
  ASSERT_DOUBLE_EQ(t.at({1, 2}), 6.0);

  const Tensor<double> &ct = t;

  ASSERT_DOUBLE_EQ(ct.at({0, 1}), 2.0);
}

TEST(TensorTestDouble, GetLinearIndex2D_ND) {

  Tensor<double> t({2, 3}, Layout::kNd);

  ASSERT_EQ(t.get_linear_index({0, 0}), 0);
  ASSERT_EQ(t.get_linear_index({0, 2}), 2);
  ASSERT_EQ(t.get_linear_index({1, 0}), 2);
  ASSERT_EQ(t.get_linear_index({1, 2}), 4);
}

TEST(TensorTestDouble, GetLinearIndex4D_NCHW) {

  Tensor<double> t({2, 3, 4, 5}, Layout::kNchw);

  ASSERT_EQ(t.get_linear_index({0, 0, 0, 0}), 0);
  ASSERT_EQ(t.get_linear_index({1, 2, 3, 4}), 119);
}

TEST(TensorTestDouble, GetLinearIndex4D_NHWC) {

  Tensor<double> t({2, 3, 4, 5}, Layout::kNhwc);

  ASSERT_EQ(t.get_linear_index({0, 0, 0, 0}), 0);
  ASSERT_EQ(t.get_linear_index({1, 2, 3, 4}), 119);
}
TEST(TensorTestDouble, GetLinearIndex4D_ND) {

  Tensor<double> t4d_nd({2, 3, 4, 5}, Layout::kNd);

  ASSERT_EQ(t4d_nd.get_linear_index({0, 0, 0, 0}), 0);
  ASSERT_EQ(t4d_nd.get_linear_index({1, 2, 3, 4}), 119);
}

TEST(TensorTestDouble, GetLinearIndex_OutOfBounds) {

  Tensor<double> t2d({2, 3}, Layout::kNd);
  Tensor<double> t4d_nchw({2, 3, 4, 5}, Layout::kNchw);
  Tensor<double> t4d_nhwc({2, 3, 4, 5}, Layout::kNhwc);
  Tensor<double> t4d_nd({2, 3, 4, 5}, Layout::kNd);

  EXPECT_THROW(t2d.get_linear_index({2, 0}), std::out_of_range);
  EXPECT_THROW(t2d.get_linear_index({0, 3}), std::out_of_range);
  EXPECT_THROW(t4d_nchw.get_linear_index({2, 0, 0, 0}), std::out_of_range);
  EXPECT_THROW(t4d_nhwc.get_linear_index({0, 3, 0, 0}), std::out_of_range);
  EXPECT_THROW(t4d_nd.get_linear_index({0, 0, 4, 0}), std::out_of_range);
}

TEST(TensorTestDouble, GetLinearIndex_WrongNumberOfIndices) {
  Tensor<double> t2d({2, 3}, Layout::kNd);
  Tensor<double> t4d_nchw({2, 3, 4, 5}, Layout::kNchw);

  EXPECT_THROW(t2d.get_linear_index({0}), std::runtime_error);
  EXPECT_THROW(t4d_nchw.get_linear_index({0, 0, 0}), std::runtime_error);
}