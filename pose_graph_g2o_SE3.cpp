#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include <Eigen/Core>

using namespace std;

/************************************************
 * 本程序演示如何用g2o solver进行位姿图优化
 * sphere.g2o是人工生成的一个Pose graph，我们来优化它。
 * 尽管可以直接通过load函数读取整个图，但我们还是自己来实现读取代码，以期获得更深刻的理解
 * 这里使用g2o/types/slam3d/中的SE3表示位姿，它实质上是四元数而非李代数.
 * **********************************************/

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage: pose_graph_g2o_SE3 sphere.g2o 0.001 1.0" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    float lambda = atof(argv[2]);
    float ratio = atof(argv[3]);

    Eigen::MatrixXd info_mat = Eigen::MatrixXd::Identity(6, 6);
    for (int i = 3; i < 6; ++i) {
        info_mat(i, i) = ratio;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出


    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    Eigen::Matrix3d last_rotation;
    Eigen::Vector3d last_transition;
    while (!fin.eof()) {
        // 读取文件数据
        Eigen::MatrixXd pose(3, 4);
        Eigen::Matrix3d rotation;
        Eigen::Vector3d transition;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                fin >> pose(i, j);
            }
        }
        rotation = pose.block<3, 3>(0, 0);
        transition = pose.block<3, 1>(0, 3);
        
        // SE3 顶点
        g2o::VertexSE3 *v = new g2o::VertexSE3();
        v->setId(vertexCnt++);
        v->setEstimate(g2o::SE3Quat(rotation, transition));
        if (vertexCnt == 1)
            v->setFixed(true);
        optimizer.addVertex(v);

        // SE3-SE3 边
        if (vertexCnt != 1) {
            g2o::EdgeSE3 *e = new g2o::EdgeSE3();
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[vertexCnt - 2]);
            e->setVertex(1, optimizer.vertices()[vertexCnt - 1]);
            Eigen::Matrix3d dr = last_rotation.transpose() * rotation;
            Eigen::Vector3d dt = last_rotation.transpose() * (transition - last_transition);
            e->setMeasurement(g2o::SE3Quat(dr, dt));
            e->setInformation(info_mat);
            optimizer.addEdge(e);
        }

        last_rotation = rotation;
        last_transition = transition;

        if (!fin.good()) break;
    }

    // SE3-SE3 回环边
    g2o::EdgeSE3 *e = new g2o::EdgeSE3();
    e->setId(edgeCnt++);
    e->setVertex(0, optimizer.vertices()[vertexCnt - 1]);
    e->setVertex(1, optimizer.vertices()[0]);
    e->setMeasurement(g2o::SE3Quat(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()));
    e->setInformation(info_mat * lambda);
    optimizer.addEdge(e);

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);

    g2o::VertexSE3* v = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(vertexCnt - 1));
    Eigen::Isometry3d pose = v->estimate();
    cout << "Pose=" << endl << pose.matrix() << endl;

    // 保存优化结果
    ofstream fout("refine_pose.txt");
    for (int i = 0; i < vertexCnt; ++i) {
        g2o::VertexSE3* v = dynamic_cast<g2o::VertexSE3*>(optimizer.vertex(i));
        Eigen::Isometry3d pose = v->estimate();
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                fout << pose(r, c) << " ";
            }
        }
        fout << "\n";
    }

    return 0;
}
