#include "./layer/ElementwiseLayer.cpp"
#include "./tensor/tensor.h"
#include "gtest/gtest.h"

TEST(ElementwiseLayerMockTest, configure_success_matching_shapes) {
  ElementwiseLayerMock layer(100, ElementwiseOp::ADD);
  Shape shape1({10, 20, 3});
  Shape shape2({10, 20, 3});
  Shape output_shape_ref;

  ASSERT_NO_THROW(layer.configure(shape1, shape2, output_shape_ref));
  EXPECT_EQ(output_shape_ref.dimensions, shape1.dimensions);
  EXPECT_EQ(layer.get_output_shape().dimensions, shape1.dimensions);
}

TEST(ElementwiseLayerMockTest, configure_fail_shape_mismatch) {
  ElementwiseLayerMock layer(101, ElementwiseOp::ADD);
  Shape shape1({10, 20, 3});
  Shape shape2({10, 20, 4});
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(shape1, shape2, output_shape_ref),std::runtime_error);
}

TEST(ElementwiseLayerMockTest, configure_fail_rank_mismatch) {
  ElementwiseLayerMock layer(102, ElementwiseOp::ADD);
  Shape shape1({10, 20, 3});
  Shape shape2({10, 20});
  Shape output_shape_ref;

  EXPECT_THROW(layer.configure(shape1, shape2, output_shape_ref),std::runtime_error);
}

TEST(ElementwiseLayerMockTest, exec_before_configure_fail) {
  ElementwiseLayerMock layer(103, ElementwiseOp::ADD);
  Tensor<double> input(Shape({1, 1, 1}));
  Tensor<double> output(Shape({1, 1, 1}));

  EXPECT_THROW(layer.exec(input, output), std::runtime_error);
}

TEST(ElementwiseLayerMockTest, get_output_shape_before_configure_fail) {
  ElementwiseLayerMock layer(104, ElementwiseOp::ADD);

  EXPECT_THROW(layer.get_output_shape(), std::runtime_error);
}

TEST(ElementwiseLayerMockTest, exec_success_after_configure) {
  ElementwiseLayerMock layer(105, ElementwiseOp::ADD);
  Shape s({5, 5});
  Shape out_ref;

  layer.configure(s, s, out_ref);
  Tensor<double> t_in(s);
  Tensor<double> t_out(out_ref);

  ASSERT_NO_THROW(layer.exec(t_in, t_out));
}

TEST(ElementwiseLayerMockTest, get_type_name_returns_correct_name_for_add) {
  ElementwiseLayerMock layer(106, ElementwiseOp::ADD);

  EXPECT_EQ(layer.get_type_name(), "ElementwiseAddLayerMock");
}

TEST(ElementwiseLayerMockTest, get_type_name_returns_correct_name_for_mul) {
  ElementwiseLayerMock layer(107, ElementwiseOp::MUL);

  EXPECT_EQ(layer.get_type_name(), "ElementwiseMulLayerMock");
}