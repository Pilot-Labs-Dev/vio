//
// Created by root on 9/18/21.
//

#ifndef VIO_BUNDLE_ADJUSTMENT_H
#define VIO_BUNDLE_ADJUSTMENT_H
#include <vio/global.h>

namespace g2o {
    class EdgeProjectXYZ2UV;
    class SparseOptimizer;
    class VertexSE2Expmap;
    class VertexSBAPointXYZ;
}

namespace vio {

    typedef EdgeProjectXYZ2UV g2oEdgeSE3;
    typedef g2o::VertexSE2Expmap g2oFrameSE2;
    typedef g2o::VertexSBAPointXYZ g2oPoint;

    class Frame;
    class Point;
    class Feature;
    class Map;

/// Local, global and 2-view bundle adjustment with g2o

/// Temporary container to hold the g2o edge with reference to frame and point.
    struct EdgeContainerSE3{
        std::shared_ptr<g2oEdgeSE3>     edge;
        std::shared_ptr<Frame>          frame;
        std::shared_ptr<Feature>        feature;
        bool            is_deleted;
        EdgeContainerSE3(std::shared_ptr<g2oEdgeSE3> e, std::shared_ptr<Frame> frame, std::shared_ptr<Feature> feature) :
        edge(e), frame(frame), feature(feature), is_deleted(false)
        {}
    };

/// Optimize two camera frames and their observed 3D points.
/// Is used after initialization.
    void twoViewBA(std::shared_ptr<Frame> frame1, std::shared_ptr<Frame> frame2, double reproj_thresh, Map* map);

/// Local bundle adjustment.
/// Optimizes core_kfs and their observed map points while keeping the
/// neighbourhood fixed.
    void localBA(
            std::shared_ptr<Frame> center_kf,
            set<FramePtr>* core_kfs,
            Map* map,
            size_t& n_incorrect_edges_1,
            size_t& n_incorrect_edges_2,
            double& init_error,
            double& final_error);

/// Global bundle adjustment.
/// Optimizes the whole map. Is currently not used in VIO.
    void globalBA(Map* map);

/// Initialize g2o with solver type, optimization strategy and camera model.
    void setupG2o(g2o::SparseOptimizer * optimizer);

/// Run the optimization on the provided graph.
    void runSparseBAOptimizer(
            g2o::SparseOptimizer* optimizer,
            unsigned int num_iter,
            double& init_error,
            double& final_error);

/// Create a g2o vertice from a keyframe object.
    std::shared_ptr<g2oFrameSE2> createG2oFrameSE2(
            FramePtr kf,
            size_t id,
            bool fixed);

/// Creates a g2o vertice from a mappoint object.
    g2oPoint* createG2oPoint(
            Vector3d pos,
            size_t id,
            bool fixed);

/// Creates a g2o edge between a g2o keyframe and mappoint vertice with the provided measurement.
    std::shared_ptr<g2oEdgeSE3> createG2oEdgeSE3(
            std::shared_ptr<g2oFrameSE2> v_kf,
            std::shared_ptr<g2oPoint> v_mp,
            const Vector2d& f_up,
            bool robust_kernel,
            double huber_width,
            double weight = 1);

} // namespace VIO


#endif //VIO_BUNDLE_ADJUSTMENT_H
