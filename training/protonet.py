### Not Ready ###


class PrototypicalNet(nn.Module):
    def __init__(self, use_gpu=True, Data_Vector_Length=100, ModelSelect="FCN"):
        super(PrototypicalNet, self).__init__()

        self.Data_Vector_Length = Data_Vector_Length

        if ModelSelect == "RNN":
            self.f = RNN()
        elif ModelSelect == "FCN":
            if self.Data_Vector_Length == 50:
                self.f = Net50()
            elif self.Data_Vector_Length == 100:
                self.f = Net100()
            else:
                self.f = Net200()
        # self.f = Net()
        self.gpu = use_gpu
        if self.gpu:
            self.f = self.f.cuda()

    def forward(self, datax, datay, Ns, Nc, Nq, total_classes):

        k = total_classes.shape[0]
        K = np.random.choice(total_classes, Nc, replace=False)
        Query_x = torch.Tensor()
        if self.gpu:
            Query_x = Query_x.cuda()
        Query_y = []
        Query_y_count = []
        centroid_per_class = {}
        class_label = {}
        label_encoding = 0
        for cls in K:
            S_cls, Q_cls = self.random_sample_cls(datax, datay, Ns, Nq, cls)
            centroid_per_class[cls] = self.get_centroid(S_cls, Nc)
            # print(centroid_per_class[cls])
            class_label[cls] = label_encoding
            label_encoding += 1
            # Joining all the query set together
            Query_x = torch.cat((Query_x, Q_cls), 0)
            Query_y += [cls]
            Query_y_count += [Q_cls.shape[0]]
        Query_y, Query_y_labels = self.get_query_y(Query_y, Query_y_count, class_label)
        Query_x = self.get_query_x(Query_x, centroid_per_class, Query_y_labels)
        return Query_x, Query_y, self.f

    def random_sample_cls(self, datax, datay, Ns, Nq, cls):
        """
        Randomly samples Ns examples as support set and Nq as Query set
        """

        datay = datay.numpy()
        data = datax[(np.nonzero(datay == cls))[0]]

        perm = torch.randperm(data.shape[0])
        idx = perm[:Ns]
        S_cls = data[idx]
        idx = perm[Ns : Ns + Nq]
        Q_cls = data[idx]
        # print(True in np.isnan(S_cls.cpu().float()))
        # print(True in torch.isnan(S_cls))
        if self.gpu:
            S_cls = S_cls.cuda()
            Q_cls = Q_cls.cuda()
        # print(True in torch.isnan(S_cls))
        # print(True in np.isnan(S_cls.cpu().float()))
        return S_cls.float(), Q_cls.float()

    def get_centroid(self, S_cls, Nc):
        """
        Returns a centroid vector of support set for a class
        """
        S_cls = S_cls.cuda()
        embed = self.f(S_cls.float())
        return torch.sum(embed, 0).unsqueeze(1).transpose(0, 1) / Nc

    def get_query_y(self, Qy, Qyc, class_label):
        """
        Returns labeled representation of classes of Query set and a list of labels.
        """
        labels = []
        m = len(Qy)
        for i in range(m):
            labels += [Qy[i]] * Qyc[i]
        labels = np.array(labels).reshape(len(labels), 1)
        label_encoder = LabelEncoder()
        # Query_y = torch.Tensor(labels).int().squeeze()
        Query_y = torch.Tensor(label_encoder.fit_transform(labels).astype(int)).long()
        if self.gpu:
            Query_y = Query_y.cuda()
        Query_y_labels = np.unique(labels)
        return Query_y, Query_y_labels

    def get_centroid_matrix(self, centroid_per_class, Query_y_labels):
        """
        Returns the centroid matrix where each column is a centroid of a class.
        """
        centroid_matrix = torch.Tensor()
        if self.gpu:
            centroid_matrix = centroid_matrix.cuda()
        for label in Query_y_labels:
            centroid_matrix = torch.cat((centroid_matrix, centroid_per_class[label]))
        if self.gpu:
            centroid_matrix = centroid_matrix.cuda()
        # print('get_query_x centroid_matrix')
        # print(centroid_matrix)
        return centroid_matrix

    def get_query_x(self, Query_x, centroid_per_class, Query_y_labels):
        """
        Returns distance matrix from each Query image to each centroid.
        """
        centroid_matrix = self.get_centroid_matrix(centroid_per_class, Query_y_labels)

        # print(centroid_matrix)
        # embeding: net
        # print('Query_x')
        Query_x = self.f(Query_x)
        # print(Query_x)
        # print(Query_x[17,:])
        m = Query_x.size(0)
        n = centroid_matrix.size(0)
        # The below expressions expand both the matrices such that they become compatible to each other in order to caclulate L2 distance.
        # Expanding centroid matrix to "m".
        centroid_matrix = centroid_matrix.expand(
            m, centroid_matrix.size(0), centroid_matrix.size(1)
        )
        Query_matrix = Query_x.expand(n, Query_x.size(0), Query_x.size(1)).transpose(
            0, 1
        )  # Expanding Query matrix "n" times
        Temp_A = centroid_matrix.transpose(1, 2)
        Temp_B = Query_matrix.transpose(1, 2)
        # Qx = torch.pairwise_distance(centroid_matrix.transpose(
        #     1, 2), Query_matrix.transpose(1, 2))
        Qx = torch.cosine_similarity(
            centroid_matrix.transpose(1, 2), Query_matrix.transpose(1, 2), dim=1
        )
        # print('Qx')
        # print(Qx)
        return Qx
