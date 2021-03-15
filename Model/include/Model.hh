/**
 * @file
 */

#include <atomic>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#ifndef KVCG_MODEL_HH
#define KVCG_MODEL_HH

namespace pt = boost::property_tree;

namespace kvgpu {

    template<typename K>
    struct Model {
        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        virtual bool operator()(K key, unsigned hash) const = 0;
    };

    template<typename K>
    struct SimplModel : public Model<K> {

        SimplModel() : value(10000000) {

        }

        explicit SimplModel(int v) : value(v) {

        }

        explicit SimplModel(const std::string &filename) {
            pt::ptree root;
            pt::read_json(filename, root);
            value = root.get<int>("value");
        }

        SimplModel(const SimplModel<K> &other) {
            value = other.value;
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return hash < value;
        }

    private:
        int value;
    };

    template<typename K>
    struct AnalyticalModel : public Model<K> {

        AnalyticalModel() : threshold(1.5e-5), size(100000), pred(new std::atomic<float *>()) {
            *pred = new float[size];
            for (int i = 0; i < size; i++) {
                (*pred)[i] = 1.0 / size;
            }
        }

        explicit AnalyticalModel(float v) : threshold(v), size(100000), pred(new std::atomic<float *>()) {
            *pred = new float[size];
            for (int i = 0; i < size; i++) {
                (*pred)[i] = 1.0 / size;
            }
        }

        explicit AnalyticalModel(const std::string &filename) : pred(new std::atomic<float *>()) {
            pt::ptree root;
            pt::read_json(filename, root);
            threshold = root.get<int>("threshold");
            size = root.get<size_t>("size");
            *pred = new float[size];

            for (int i = 0; i < size; i++) {
                (*pred)[i] = root.get<float>("pred" + std::to_string(i));
            }

        }

        AnalyticalModel(const AnalyticalModel<K> &other) {
            threshold = other.threshold;
            size = other.size;
            pred = other.pred;
        }

        void train(std::vector <std::pair<K, unsigned>> v) {
            for (int i = 0; i < size; i++) {
                (*pred)[i] = 0.0;
            }
            for (auto &elm : v) {
                (*pred)[elm.second % size]++;
            }
            for (int i = 0; i < size; i++) {
                (*pred)[i] /= v.size();
            }
        }

        void persist(const std::string &filename) const {
            pt::ptree root;

            root.put("threshold", threshold);
            root.put("size", size);
            for (int i = 0; i < size; i++) {
                root.put("pred" + std::to_string(i), (*pred)[i]);
            }

            pt::write_json(filename, root);
        }


        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return (*pred)[hash % size] >= threshold;
        }

    private:
        float threshold;
        size_t size;
        std::atomic<float *> *pred;
    };

}

#endif //KVCG_MODEL_HH
