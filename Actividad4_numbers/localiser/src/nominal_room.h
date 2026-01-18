#pragma once
#include <QPointF>
#include <QRectF>
#include <Eigen/Dense>
#include <vector>
#include <cppitertools/enumerate.hpp>
#include <cppitertools/sliding_window.hpp>

#include "common_types.h"

  struct NominalRoom
        {
            float width; //  mm
            float length;
            Doors doors;
            bool visited = false;
            explicit NominalRoom(const float width_=10000.f, const float length_=5000.f, Corners  corners_ = {}) :
                width(width_), length(length_)
            {};
            [[nodiscard]] Corners corners() const
            {
                // compute corners from width and length
                return {
                    {QPointF{-width/2.f, -length/2.f}, 0.f, 0.f},
                    {QPointF{width/2.f, -length/2.f}, 0.f, 0.f},
                    {QPointF{width/2.f, length/2.f}, 0.f, 0.f},
                    {QPointF{-width/2.f, length/2.f}, 0.f, 0.f}
                };
            }

            [[nodiscard]] Walls get_walls() const
            {
                auto cs = corners();
                cs.push_back(cs[0]);
                Walls walls;
                for (const auto &[i,c] : cs | iter::sliding_window(2) | iter::enumerate)
                {
                    const auto &[p1, _, __] = c[0];
                    const auto &[p2, ___, ____] = c[1];
                    Eigen::Vector2f p1_v(p1.x(), p1.y());
                    Eigen::Vector2f p2_v(p2.x(), p2.y());
                    walls.emplace_back(Eigen::ParametrizedLine<float, 2>::Through(p1_v, p2_v), i, p1, p2);
                }
                return walls;
            }
      // Call get_walls(). Use std::ranges::min_element(walls, [p](w1, w2){ return w1.distance(p1) < w2.distance(p2)}) to select the wall that is closest to p.

             [[nodiscard]] Wall get_closest_wall_to_point(const Eigen::Vector2f &p )
             {
                 const Walls walls = get_walls();
                if (walls.empty()) return {};
                 const auto m = std::ranges::min_element(walls, [p](auto &w1, auto &w2)
                 {
                     return std::get<0>(w1).distance(p) < std::get<0>(w2).distance(p);
                 });

                return *m;
             }

            [[nodiscard]] Eigen::Vector2f get_projection_of_point_on_closest_wall(const Eigen::Vector2f &p)
            {
                const auto wall = get_closest_wall_to_point(p);
                return std::get<0>(wall).projection(p);
            }

            [[nodiscard]] QRectF rect() const
            {
                return QRectF{-width/2.f, -length/2.f, width, length};
            }
            [[nodiscard]] Corners transform_corners_to(const Eigen::Affine2d &transform) const  // for room to robot pass the inverse of robot_pose
            {
                Corners transformed_corners;
                for(const auto &[p, _, __] : corners())
                {
                    auto ep = Eigen::Vector2d{p.x(), p.y()};
                    Eigen::Vector2d tp = transform * ep;
                    transformed_corners.emplace_back(QPointF{static_cast<float>(tp.x()), static_cast<float>(tp.y())}, 0.f, 0.f);
                }
                return transformed_corners;
            }
        };